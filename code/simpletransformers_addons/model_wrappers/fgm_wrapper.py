import logging
import torch

from simpletransformers_addons.model_wrappers.proxy import Proxy

logger = logging.getLogger(__name__)

class FGM(object):
    """Reference: https://arxiv.org/pdf/1605.07725.pdf"""
    def __init__(self,
                 model,
                 emb_name='word_embeddings.',
                 epsilon=1.0):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self):
        """Add adversity."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        """restore embedding"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if self.emb_name in name:
                    param.grad = self.grad_backup[name]
                else:
                    param.grad += self.grad_backup[name]


class FGMInnerModelWrapper(Proxy):
    """transformers model wrapper."""
    def __init__(self, model,
                 emb_name='word_embeddings.',
                 epsilon=1.0,
                 n_gpu=1,
                 fp16=False,
                 gradient_accumulation_steps=1):
        super(FGMInnerModelWrapper, self).__init__(model)
        self._model = model
        self._fgm = FGM(model,
                        emb_name, epsilon)
        self._scaler = None
        self._n_gpu = n_gpu
        self._fp16 = fp16
        self._gradient_accumulation_steps = gradient_accumulation_steps

    def __call__(self, *args, **kwargs):
        logging.debug('fgm call model...')
        outputs = self._model(*args, **kwargs)
        if not self.training:
            return outputs
        if self._fp16:
            from torch.cuda import amp
            with amp.autocast(enabled=False):
                self._run_inner(outputs, *args, **kwargs)
        else:
            self._run_inner(outputs, *args, **kwargs)
        loss = outputs[0]
        # simulate loss
        loss = loss.detach()
        dummy_var = torch.zeros((), requires_grad=True)
        loss = loss + dummy_var - dummy_var
        return (loss, ) + outputs[1:]

    def _get_loss(self, *args, **kwargs):
        if self._fp16:
            from torch.cuda import amp
            with amp.autocast():
                outputs = self._model(*args, **kwargs)
                loss = outputs[0]
        else:
            outputs = self._model(*args, **kwargs)
            loss = outputs[0]
        if self._n_gpu > 1:
            loss = loss.mean()
        if self._gradient_accumulation_steps > 1:
            loss = loss / self._gradient_accumulation_steps
        return loss

    def _backward(self, loss):
        if self._fp16:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def _run_inner(self, outputs, *args, **kwargs):
        loss = outputs[0]
        if self._n_gpu > 1:
            loss = loss.mean()   # mean() to average on multi-gpu parallel training
        if self._gradient_accumulation_steps > 1:
            loss = loss / self._gradient_accumulation_steps
        # normal backward
        self._backward(loss)
        self._fgm.backup_grad()
        # 对抗训练
        self._fgm.attack()
        self._model.zero_grad()
        loss_adv = self._get_loss(*args, **kwargs)
        self._backward(loss_adv)
        self._fgm.restore_grad()
        self._fgm.restore()


class FGMWrapper(Proxy):
    """simpletransformers model wrapper."""
    def __init__(self, model,
                 emb_name='word_embeddings.',
                 epsilon=1.0,
                 alpha=0.3,
                 attack_number=3):
        super(FGMWrapper, self).__init__(model)
        self._model = FGMInnerModelWrapper(model.model,
                                          emb_name=emb_name,
                                          epsilon=epsilon,
                                          n_gpu=self.args.n_gpu,
                                          fp16=self.args.fp16,
                                          gradient_accumulation_steps=self.args.gradient_accumulation_steps)
        model.model = self._model

    def train_model(self, *args, **kwargs):
        # patch for fp16
        outer_models = [self._model]
        if self.args.fp16:
            from torch.cuda import amp
            class GradScaler(amp.GradScaler):
                def __init__(self, *args, **kwargs):
                    logging.debug('call patched grad scaler')
                    super(GradScaler, self).__init__(*args, **kwargs)
                    outer_models[0]._scaler = self
            amp.GradScaler = GradScaler
        self._obj.train_model(*args, **kwargs)

