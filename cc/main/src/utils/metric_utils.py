# MRR metric used for Code Complition task
import torch


def _mrr(pred, y_true, oov, k=10):
    """
        Mean reciprocal rank.
        pred: [bs * L, N],
        y_true: [bs * L]
    """
    if pred.size(-1) < k:
        _, pred = torch.topk(pred, k=pred.size(-1),
                             dim=-1)  # hot fix for the case of a very small vocabulary and pointer
    else:
        _, pred = torch.topk(pred, k=k, dim=-1)
    pred = pred.cpu()
    y_true = y_true.cpu()
    pred = (pred == y_true[:, None])
    pred &= ~(y_true[:, None] == oov)  # Out of Vocab predictions get zero score
    r = torch.nonzero(pred, as_tuple=True)[1]
    if len(r) == 0:
        return torch.tensor(0.0, device=y_true.device), torch.tensor(0, device=y_true.device)
    ln = y_true.numel()
    return torch.tensor((1.0 / (r + 1.0)).sum(), device=y_true.device), torch.tensor(ln, device=y_true.device)


def mrr(y_pred, y, ext, vocab, use_pointer=False):
    """
    y: Tensor [bs, L]
    pred: Tensor [bs, L, N]
    ext: Tensor [bs]
    """
    ext = ext.unsqueeze(-1).repeat(1, y.size(-1))
    ext_ids = torch.arange(y.size(-1), device=ext.device).view(1, -1).repeat(*(y.size()[:-1] + (1,)))
    where = ext_ids >= ext
    where &= y != vocab.pad_idx  # calc loss only on known tokens and filter padding
    where &= y != vocab.empty_idx
    where = where.view(-1)

    y_pred = y_pred.view(-1, y_pred.size(-1))
    y = y.view(-1)
    metric_sum, ln = _mrr(y_pred[where], y[where], vocab.unk_idx)

    pred_acc = (torch.argmax(y_pred[where], dim=-1) == y[where]).int().sum().to(y_pred.device)

    return metric_sum, ln, pred_acc, ln


def acc(types, types_true, values, values_true, ext, types_vocab, values_vocab):
    """
    types,values:[bs, L, N]
    types_true,values_true:[bs, L]
    ext: Tensor [bs]
    """
    assert types.shape[:1] == values.shape[:1]
    assert types_true.shape == values_true.shape

    ext = ext.unsqueeze(-1).repeat(1, types_true.size(-1))
    ext_ids = torch.arange(types_true.size(-1), device=ext.device).view(1, -1).repeat(*(types_true.size()[:-1] + (1,)))
    where = ext_ids >= ext

    where &= types_true != types_vocab.pad_idx  # calc loss only on known tokens and filter padding
    where &= values_true != values_vocab.pad_idx  # calc loss only on known tokens and filter padding
    # where &= types_true != types_vocab.empty_idx
    # where &= values_true != values_vocab.empty_idx

    where = where.view(-1)

    types = types.view(-1, types.size(-1))
    types_true = types_true.view(-1)
    values = values.view(-1, values.size(-1))
    values_true = values_true.view(-1)

    _types, _value = types[where], values[where]  # bs*l,N
    _type_true, _value_true = types_true[where], values_true[where]  # bs*l,

    _types_idx, _value_idx = torch.argmax(_types, dim=-1), torch.argmax(_value, dim=-1)  # bs*l

    type_count = (_types_idx == _type_true)
    value_count = (_value_idx == _value_true)

    type_count = type_count.masked_fill(_type_true == types_vocab.unk_idx, False)
    value_count = value_count.masked_fill(_value_true == values_vocab.unk_idx, False)

    #########################
    # print('--------------')
    # print(len(type_count))
    # print(type_count.int().sum())

    type_count = type_count.masked_fill(_type_true == types_vocab.empty_idx, True)

    # print(type_count.int().sum())

    # print(value_count.int().sum())
    value_count = value_count.masked_fill(_value_true == values_vocab.empty_idx, True)
    # print(value_count.int().sum())

    ###############
    pair_count = (type_count & value_count).int()
    return pair_count.sum().to(types.device), torch.tensor(len(type_count), device=types.device)
