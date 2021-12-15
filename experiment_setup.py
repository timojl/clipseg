from torch.functional import Tensor
from general_utils import get_attribute, load_model

import torch
import inspect
from general_utils import filter_args, AttributeDict
from general_utils import TrainingLogger
import json
import math

from general_utils import log

import numpy as np
from functools import partial
from os.path import expanduser, join, isfile, basename

from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from contextlib import nullcontext
from torch.utils.data import DataLoader


DATASET_CACHE = dict()
CLASSIC_MODELS = {'ConditionBase4', 'PFENetWrapper', 'HSNetWrapper'}

pascal_classes = {a['id']: a['synonyms'] for a in json.load(open('datasets/pascal_classes.json'))}


def validate(model, dataset, config):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    metric_class, use_metric = config.val_metric_class, config.use_val_metric
    loss_fn = get_attribute(config.loss)

    model.eval()
    model.cuda()

    if metric_class is not None:
        metric = get_attribute(metric_class)()

    with torch.no_grad():

        i, losses = 0, []
        for data_x, data_y in data_loader:

            data_x = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_x]
            data_y = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_y]

            if model.__class__.__name__ in CLASSIC_MODELS:
                pred, = model(data_x[0], data_x[1], data_x[2])
                visual_q = None
            else:
                prompts = model.sample_prompts(data_x[1], prompt_list=('a photo of a {}',))
                pred, visual_q, _, _  = model(data_x[0], prompts, return_features=True)

            if metric_class is not None:
                metric.add([pred], data_y)

            # pred = model(data_x[0], prompts)
            # loss = loss_fn(pred[0], data_y[0])
            loss = loss_fn(pred, data_y[0])
            losses += [float(loss)]

            i += 1

            if config.val_max_iterations is not None and i > config.val_max_iterations:
                break


    if use_metric is None:
        return np.mean(losses), {}, False
    else:
        metric_scores = {m: s for m, s in zip(metric.names(), metric.value())} if metric is not None else {}
        return np.mean(losses), metric_scores, True


def cosine_warmup_lr(i, warmup=10, max_iter=90):
    """ Cosine LR with Warmup """
    if i < warmup:
        return (i+1)/(warmup+1)
    else:
        return 0.5 + 0.5*math.cos(math.pi*(((i-warmup)/(max_iter- warmup))))


def train_loop(config):

    from general_utils import log

    config = AttributeDict(config)

    val_interval, best_val_loss, best_val_score = config.val_interval, float('inf'), float('-inf')

    model_cls = get_attribute(config.model)
    _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)
    model = model_cls(**model_args).cuda()
 
    if config.weight:
        # use weight and don't train
        weights = torch.load(expanduser(config.weight))
        model.load_state_dict(weights)
        return model

    if config.init_weights:
        # init weights and train
        weights = torch.load(expanduser(config.init_weights))
        
        strict = model.__class__.__name__ in CLASSIC_MODELS
        model.load_state_dict(weights, strict=strict)

    dataset_cls = get_attribute(config.dataset)
    _, dataset_args, _ = filter_args(config, inspect.signature(dataset_cls).parameters)

    dataset = dataset_cls(**dataset_args)

    log.info(f'Train dataset {dataset.__class__.__name__} (length: {len(dataset)})')

    if val_interval is not None:
        dataset_val_args = {k[4:]: v for k,v in config.items() if k.startswith('val_') and k != 'val_interval'}
        _, dataset_val_args, _ = filter_args(dataset_val_args, inspect.signature(dataset_cls).parameters)
        print('val args', {**dataset_args, **{'split': 'val', 'aug': 0}, **dataset_val_args})
    
        dataset_val = dataset_cls(**{**dataset_args, **{'split': 'val', 'aug': 0}, **dataset_val_args})

    # optimizer
    opt_cls = get_attribute(config.optimizer)
    if config.optimize == 'torch.optim.SGD':
        opt_args = {'momentum': config.momentum if 'momentum' in config else 0}
    else:
        opt_args = {}
    opt = opt_cls(model.parameters(), lr=config.lr, **opt_args)

    if config.lr_scheduler == 'cosine':
        assert config.T_max is not None and config.eta_min is not None
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config.T_max, config.eta_min)
    elif config.lr_scheduler == 'warmup_cosine':        
        lr_scheduler = LambdaLR(opt, partial(cosine_warmup_lr, max_iter=(config.max_iterations), warmup=config.warmup))
    else:
        lr_scheduler = None

    batch_size, max_iterations = config.batch_size, config.max_iterations

    loss_fn = get_attribute(config.loss)

    if config.amp:
        log.info('Using AMP')
        autocast_fn = autocast
        scaler = GradScaler()
    else:
        autocast_fn, scaler = nullcontext, None

    # check for unused arguments
    config.assume_no_unused_keys(exceptions=['name', 'mix', 'mix_text_min', 'mix_text_max', 'extra_loss_tv_align', 'extra_loss_corr', 'norm_cond',
                                             'checkpoint_iterations'])        
    
    save_only_trainable = model.__class__.__name__ not in CLASSIC_MODELS

    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    # disable config when hyperparam. opt. to avoid writing logs.
    tracker_config = config if not config.hyperparameter_optimization else None

    with TrainingLogger(log_dir=config.name, interval=150, model=model, config=tracker_config, utilization_iters=500) as logger:

        i = 0
        while True:
            for data_x, data_y in data_loader:

                # between caption and output feature.
                # 1. Sample random captions
                # 2. Check alignment with CLIP

                # randomly mix text and visual support conditionals
                if config.mix:

                    assert config.mask.startswith('text_and')

                    with autocast_fn():
                        # data_x[1] = text label
                        prompts = model.sample_prompts(data_x[1])

                        # model.clip_model()

                        text_cond = model.compute_conditional(prompts)
                        if model.__class__.__name__ == 'CLIPDensePredTMasked':
                            # when mask=='separate'
                            visual_s_cond, _, _ = model.visual_forward_masked(data_x[2].cuda(), data_x[3].cuda())
                        else:
                            # data_x[2] = visual prompt
                            visual_s_cond, _, _ = model.visual_forward(data_x[2].cuda())

                    max_txt = config.mix_text_max if config.mix_text_max is not None else 1
                    batch_size = text_cond.shape[0]

                    # sample weights for each element in batch
                    text_weights = torch.distributions.Uniform(config.mix_text_min, max_txt).sample((batch_size,))[:, None]
                    text_weights = text_weights.cuda()

                    if dataset.__class__.__name__ == 'PhraseCut':
                        # give full weight to text where support_image is invalid
                        visual_is_valid = data_x[4] if model.__class__.__name__ == 'CLIPDensePredTMasked' else data_x[3]
                        text_weights = torch.max(text_weights[:,0], 1 - visual_is_valid.float().cuda()).unsqueeze(1)

                    cond = text_cond * text_weights + visual_s_cond * (1 - text_weights)

                else:
                    # no mix
                    
                    if model.__class__.__name__ == 'CLIPDensePredTMasked':
                        # compute conditional vector using CLIP masking
                        with autocast_fn():
                            assert config.mask == 'separate'
                            cond, _, _ = model.visual_forward_masked(data_x[1].cuda(), data_x[2].cuda())
                    else:
                        cond = data_x[1]
                        if isinstance(cond, torch.Tensor):
                            cond = cond.cuda()

                with autocast_fn():
                    visual_q = None
                    if model.__class__.__name__ in CLASSIC_MODELS:
                        assert config.mask == 'separate'
                        pred, = model(data_x[0].cuda(), cond, data_x[2].cuda())
                    #elif model.__class__.__name__ == 'CLIPDensePredTMasked' and config.mask == 'separate':
                    #    pred, = model(data_x[0].cuda(), data_x[1].cuda(), data_x[2].cuda(), return_features=True)                  
                    else:
                        pred, visual_q, _, _  = model(data_x[0].cuda(), cond, return_features=True)

                    loss = loss_fn(pred, data_y[0].cuda())

                    if torch.isnan(loss) or torch.isinf(loss):
                        # skip if loss is nan
                        log.warning('Training stopped due to inf/nan loss.')
                        return 

                    extra_loss = 0
                    loss += extra_loss

                opt.zero_grad()

                if scaler is None:
                    loss.backward()
                    opt.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                if lr_scheduler is not None:
                    lr_scheduler.step()
                    if i % 2000 == 0:
                        current_lr = [g['lr'] for g in opt.param_groups][0]
                        log.info(f'current lr: {current_lr:.5f} ({len(opt.param_groups)} parameter groups)')
                        
                logger.iter(i=i, loss=loss)
                i += 1

                if i >= max_iterations:

                    if not isfile(join(logger.base_path, 'weights.pth')):
                        # only write if no weights were already written
                        logger.save_weights(only_trainable=save_only_trainable)

                    logger.stop()
                    return
                    
                if config.checkpoint_iterations is not None and i in config.checkpoint_iterations:
                    logger.save_weights(only_trainable=save_only_trainable, weight_file=f'weights_{i}.pth')

                if val_interval is not None and i % val_interval == val_interval - 1:

                    val_loss, val_scores, maximize = validate(model, dataset_val, config)
                    
                    if len(val_scores) > 0:

                        score_str = f', scores: ' + ', '.join(f'{k}: {v}' for k, v in val_scores.items())
                        
                        if maximize and val_scores[config.use_val_metric] > best_val_score:
                            logger.save_weights(only_trainable=save_only_trainable)
                            best_val_score = val_scores[config.use_val_metric]

                        elif not maximize and val_scores[config.use_val_metric] < best_val_score:
                            logger.save_weights(only_trainable=save_only_trainable)
                            best_val_score = val_scores[config.use_val_metric]

                    else:
                        score_str = ''
                        # if no score is used, fall back to loss
                        if val_loss < best_val_loss:
                            logger.save_weights(only_trainable=save_only_trainable)
                            best_val_loss = val_loss
                    
                    log.info(f'Validation loss: {val_loss}' + score_str)
                    logger.iter(i=i, val_loss=val_loss, extra_loss=float(extra_loss), **val_scores)
                    model.train()

            print('epoch complete')


def score(config, train_checkpoint_id, train_config):

    from general_utils import log

    config = AttributeDict(config)
    # use training dataset and loss
    train_config = AttributeDict(json.load(open(f'logs/{train_checkpoint_id}/config.json')))

    cp_str = f'_{config.iteration_cp}' if config.iteration_cp is not None else ''

    if config.load_weights is None:
        model_cls = get_attribute(train_config['model'])

        _, model_args, _ = filter_args(train_config, inspect.signature(model_cls).parameters)

        model_args = {**model_args, **{k: config[k] for k in ['process_cond', 'fix_shift'] if k in config}}

        model = load_model(train_checkpoint_id, strict=model_cls.__name__ == 'ConditionBase4', model_args=model_args, 
                           weights_file=f'weights{cp_str}.pth', )
    else:
        model_cls = get_attribute(config['model'])
        _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)
        model = model_cls(**model_args)

        model.load_state_dict(torch.load(expanduser(config.load_weights)))

    model.eval()
    model.cuda()

    metric_args = dict()

    if 'threshold' in config:
        if config.metric.split('.')[-1] == 'BinaryIoU': 
            metric_args['thresholds'] = (config.threshold, 0.5)
        elif config.metric.split('.')[-1] == 'SkLearnMetrics':
            metric_args['threshold'] = config.threshold

    if 'resize_to' in config:
        metric_args['resize_to'] = config.resize_to

    if 'sigmoid' in config:
        metric_args['sigmoid'] = config.sigmoid    

    if 'custom_threshold' in config:
        metric_args['custom_threshold'] = config.custom_threshold     

    if config.test_dataset == 'pascal':
        from datasets.pfe_dataset import PFEPascalWrapper

        loss_fn = get_attribute(train_config.loss)

        shift = config.shift if 'shift' in config else 0
        splits = config.splits if 'splits' in config else [0,1,2,3]

        log.info('Test on these splits', splits)
        scores = dict()
        for split in splits:

            try:
                dataset =  DATASET_CACHE[(split, config.image_size, config.label_support)]
            except KeyError:
                print('Using split', split)
                dataset = PFEPascalWrapper(mode='val', split=split, mask=config.mask, image_size=config.image_size, label_support=config.label_support)
                # DATASET_CACHE[(split, config.image_size, config.label_support)] = dataset
            loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

            print(config.image_size, config.mask)

            log.info('Dataset length:', len(dataset))

            assert config.batch_size is None or config.batch_size == 1, 'When PFE Dataset is used, batch size must be 1'

            metric = get_attribute(config.metric)(resize_pred=True, **metric_args)

            with torch.no_grad():

                i, losses = 0, []
                for i_all, (data_x, data_y) in enumerate(loader):

                    data_x = [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_x]
                    data_y = [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_y]

                    if config.mask == 'separate':  # for old CondBase model
                        pred, = model(data_x[0], data_x[1], data_x[2])
                    else:
                        # assert config.mask in {'text', 'highlight'}
                        pred, _, _, _  = model(data_x[0], data_x[1], return_features=True)

                    # loss = loss_fn(pred, data_y[0])
                    metric.add(pred.unsqueeze(1) + shift, data_y)

                    # losses += [float(loss)]

                    i += 1
                    if config.max_iterations and i >= config.max_iterations:
                        break

            #scores[split] = {m: s for m, s in zip(metric.names(), metric.value())}
            scores[split] = metric.scores()

            log.info(f'Completed split {split}')
        
        key_prefix = config['name'] if 'name' in config else 'pas'

        all_keys = set.intersection(*[set(v.keys()) for v in scores.values()])

        valid_keys = [k for k in all_keys if all(v[k] is not None and isinstance(v[k], (int, float, np.float)) for v in scores.values())]

        return {key_prefix: {k: np.mean([s[k] for s in scores.values()]) for k in valid_keys}}


    if config.test_dataset == 'coco':
        from datasets.coco_wrapper import COCOWrapper

        coco_dataset = COCOWrapper('test', fold=train_config.fold, image_size=train_config.image_size, mask=config.mask,
                                    with_class_label=True)

        log.info('Dataset length', len(coco_dataset))
        loader = DataLoader(coco_dataset, batch_size=config.batch_size, num_workers=2, shuffle=False, drop_last=False)
        
        metric = get_attribute(config.metric)(resize_pred=True, **metric_args)

        shift = config.shift if 'shift' in config else 0

        with torch.no_grad():

            i, losses = 0, []
            for i_all, (data_x, data_y) in enumerate(loader):
                data_x = [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_x]
                data_y = [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_y]

                if config.mask == 'separate':  # for old CondBase model
                    pred, = model(data_x[0], data_x[1], data_x[2])
                else:
                    # assert config.mask in {'text', 'highlight'}
                    pred, _, _, _  = model(data_x[0], data_x[1], return_features=True)

                metric.add([pred + shift], data_y)

                i += 1
                if config.max_iterations and i >= config.max_iterations:
                    break                

        key_prefix = config['name'] if 'name' in config else 'coco'      
        return {key_prefix: metric.scores()}
        #return {key_prefix: {k: v for k, v in zip(metric.names(), metric.value())}}


    if config.test_dataset == 'phrasecut':
        from datasets.phrasecut import PhraseCut

        only_visual = config.only_visual is not None and config.only_visual
        with_visual = config.with_visual is not None and config.with_visual

        dataset = PhraseCut('test', 
                            image_size=train_config.image_size,
                            mask=config.mask, 
                            with_visual=with_visual, only_visual=only_visual, aug_crop=False, 
                            aug_color=False)

        loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=2, shuffle=False, drop_last=False)
        metric = get_attribute(config.metric)(resize_pred=True, **metric_args)

        shift = config.shift if 'shift' in config else 0

        with torch.no_grad():

            i, losses = 0, []
            for i_all, (data_x, data_y) in enumerate(loader):
                data_x = [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_x]
                data_y = [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in data_y]

                pred, _, _, _  = model(data_x[0], data_x[1], return_features=True)
                metric.add([pred + shift], data_y)

                i += 1
                if config.max_iterations and i >= config.max_iterations:
                    break                

        key_prefix = config['name'] if 'name' in config else 'phrasecut'      
        return {key_prefix: metric.scores()}
        #return {key_prefix: {k: v for k, v in zip(metric.names(), metric.value())}}

    if config.test_dataset == 'pascal_zs':
        from third_party.JoEm.model.metric import Evaluator
        from third_party.JoEm.data_loader import get_seen_idx, get_unseen_idx, VOC
        from datasets.pascal_zeroshot import PascalZeroShot, CLIPSegMultiLabel, PASCAL_VOC_CLASSES_ZS

        n_unseen = train_config.remove_classes[1]

        pz = PascalZeroShot('val', n_unseen, image_size=352)
        m = CLIPSegMultiLabel(model=train_config.name).cuda()
        m.eval();

        print(len(pz), n_unseen)
        print('training removed', [c for class_set in PASCAL_VOC_CLASSES_ZS[:n_unseen // 2] for c in class_set])

        print('unseen', [VOC[i] for i in get_unseen_idx(n_unseen)])
        print('seen', [VOC[i] for i in get_seen_idx(n_unseen)])

        loader = DataLoader(pz, batch_size=8)
        evaluator = Evaluator(21, get_unseen_idx(n_unseen), get_seen_idx(n_unseen))

        for i, (data_x, data_y) in enumerate(loader):
            pred = m(data_x[0].cuda())
            evaluator.add_batch(data_y[0].numpy(), pred.argmax(1).cpu().detach().numpy())
            
            if config.max_iter is not None and i > config.max_iter: 
                break
                
        scores = evaluator.Mean_Intersection_over_Union()        
        key_prefix = config['name'] if 'name' in config else 'pas_zs'      

        return {key_prefix: {k: scores[k] for k in ['seen', 'unseen', 'harmonic', 'overall']}}


    elif config.test_dataset in {'same_as_training', 'lvis', 'affordance'}:
        loss_fn = get_attribute(train_config.loss)


        metric_cls = get_attribute(config.metric)
        metric = metric_cls(**metric_args)

        if config.test_dataset == 'same_as_training':
            dataset_cls = get_attribute(train_config.dataset)
        elif config.test_dataset == 'affordance':
            dataset_cls = get_attribute('datasets.lvis_oneshot3.LVIS_Affordance')
            dataset_name = 'aff'
        else:
            dataset_cls = get_attribute('datasets.lvis_oneshot3.LVIS_OneShot')
            dataset_name = 'lvis'

        _, dataset_args, _ = filter_args(config, inspect.signature(dataset_cls).parameters)

        dataset_args['image_size'] = train_config.image_size  # explicitly use training image size for evaluation

        log.info('init dataset', str(dataset_cls))
        dataset = dataset_cls(**dataset_args)

        log.info(f'Score on {model.__class__.__name__} on {dataset_cls.__name__}')

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)

        # explicitly set prompts
        if config.prompt == 'plain':
            model.prompt_list = ['{}']
        elif config.prompt == 'fixed':
            model.prompt_list = ['a photo of a {}.']
        elif config.prompt == 'shuffle':
            model.prompt_list = ['a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.', '{}.']
        elif config.prompt == 'shuffle_clip':
            from models.clip_prompts import imagenet_templates
            model.prompt_list = imagenet_templates

        config.assume_no_unused_keys(exceptions=['max_iterations'])

        with torch.no_grad():  # TODO: switch to inference_mode (torch 1.9)
            i, losses = 0, []
            for data_x, data_y in data_loader:

                data_x = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_x]
                data_y = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_y]

                if model.__class__.__name__ == 'ConditionBase4':
                    pred, = model(data_x[0], data_x[1], data_x[2])
                    visual_q = None
                else:
                    pred, visual_q, _, _  = model(data_x[0], data_x[1], return_features=True)

                loss = loss_fn(pred, data_y[0])

                metric.add([pred], data_y)

                losses += [float(loss)]

                i += 1
                if config.max_iterations and i >= config.max_iterations:
                    break

        scores = {m: s for m, s in zip(metric.names(), metric.value())}

        keys = set(scores.keys())
        if dataset.negative_prob > 0 and 'mIoU' in keys:
            keys.remove('mIoU')

        name_mask = dataset.mask.replace('text_label', 'txt')[:3]
        name_neg = '' if dataset.negative_prob == 0 else '_' + str(dataset.negative_prob)
        
        score_name = config.name if 'name' in config else f'{dataset_name}_{name_mask}{name_neg}'

        scores = {score_name: {k: v for k,v in scores.items() if k in keys}}
        scores[score_name].update({'test_loss': np.mean(losses)})

        return scores
    else:
        raise ValueError('invalid test dataset')

