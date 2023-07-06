import torch
from typing import Optional

def compute_confusion_matrix(
    logits: torch.Tensor,
    targets: torch.Tensor,
    activation_function: Optional[callable] = None,
    threshold: float = 0.5
) -> tuple[int, int, int, int]:
    # checking the imput values
    if not torch.is_floating_point(logits):
        raise ValueError(f"logits is not from data type floating point")
    if targets.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("targets must from data type bool or int")
    if logits.dim() != 1 or targets.dim() != 1:
        raise ValueError("logits or targets are not 1D")
    if logits.shape != targets.shape:
        raise ValueError("logits and targets have diffrent samples")
    if targets.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64) and torch.any(torch.logical_and((targets !=0), (targets !=1))):
        raise ValueError("targets have other values than 0 or 1")

    if activation_function is not None:
        logits = activation_function(logits)

    # calculate the predictions
    predictions = (logits >= threshold).to(torch.int)

    # calculat the if the prediction was true or false and when false if pos or neg
    tp = 0; fp = 0; tn = 0; fn = 0
    for i in range(len(targets)):
        if targets[i] == predictions[i] and predictions[i] == 1:
            tp = tp+1
        elif targets[i] != predictions[i] and predictions[i] == 1:
            fp = fp+1
        elif targets[i] == predictions[i] and predictions[i] == 0:
            tn = tn+1
        elif targets[i] != predictions[i] and predictions[i] == 0:
            fn = fn+1
    return tp, fn, fp, tn


if __name__ == "__main__":
    torch.manual_seed(123)
    logits = torch.rand(size=(10,)) * 10 - 5
    targets = torch.randint(low=0, high=2, size=(10,))

    tp, fn, fp, tn = compute_confusion_matrix(
        logits, targets, activation_function=torch.sigmoid)
    print(logits)
    print(targets)
    print(f"{tp=}, {fn=}, {fp=}, {tn=}")
