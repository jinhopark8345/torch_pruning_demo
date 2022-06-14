import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reference : https://pytorch.org/tutorials/intermediate/pruning_tutorial.html


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3)
        self.conv2 = nn.Conv2d(2, 4, 3)
        self.fc1 = nn.Linear(4 * 3 * 3, 2)
        self.fc2 = nn.Linear(2, 5)
        self.fc3 = nn.Linear(5, 4)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def demo_ln_structured_pruning():
    model = LeNet()

    # model.conv1 : 2 (n_channels)* 3 * 3 (kernel size) = 18 parameters
    prune.ln_structured(
        model.conv1,
        name="weight",
        amount=0.5,  # 0 ~ 1 : percentage, integer : number of pruning parameters to prune
        n=1,  # normalization parameter
        dim=0,  # dimension where you want to operate pruning
    )

    # before prune.remove, model.conv2 have additional attributes (weight_orig & weight_mask)


def demo_l1_unstructured_pruning():
    """ l1 normalize the weights and prune <amount> lowest """
    model = LeNet()

    # model.conv1 : 2 (n_channels)* 3 * 3 (kernel size) = 18 parameters
    prune.l1_unstructured(
        model.conv1,
        name="weight",
        amount=3,  # 0 ~ 1 : percentage, integer : number of pruning parameters to prune
    )

    # before prune.remove, model.conv2 have additional attributes (weight_orig & weight_mask)


def demo_random_unstructured_pruning():
    """ randomly prune """
    model = LeNet()

    prune.random_unstructured(
        model.conv1,
        name="weight",
        amount=3,
    )
    print(model.conv1.weight)
    print(model.conv1.weight_orig)


def demo_random_structured_pruning():
    """ randomly prune """
    model = LeNet()

    prune.random_structured(
        model.conv1,
        name="weight",
        amount=0.25,
        dim=0,
    )
    print(model.conv1.weight)
    print(model.conv1.weight_mask)
    print(model.conv1.weight_orig)


def demo_prune_remove():
    model = LeNet()

    module = model.fc2  # 10 values ( 2 * 5 )

    prune.random_unstructured(
        module,
        name="weight",
        amount=0.2,
    )  # 10 -> 8 values left
    print(module.weight)

    # if we make prune permanent by prune.remove, next time we prune, we
    # won't consider previously pruned weights.
    prune.remove(module, "weight")

    # if we didn't make prune permanent: 10 -> 8 -> 4
    # if we did make prune permanent by prune.remove : 10 -> ? -> 5
    prune.l1_unstructured(
        module,
        name="weight",
        amount=0.5,
    )
    print(module.weight)

    for hook in module._forward_pre_hooks.values():
        if hook._tensor_name == "weight":
            break
    print(list(hook))


def demo_layer_pruning():
    model = LeNet()
    for name, module in model.named_modules():
        # 모든 2D-conv 층의 20% 연결에 대해 가지치기 기법을 적용
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=0.2)
        # 모든 선형 층의 40% 연결에 대해 가지치기 기법을 적용
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.4)

    breakpoint()
    print(dict(model.named_buffers()).keys())  # 존재하는 모든 마스크들을 확인

    # $ (Pdb++) pp new_model.state_dict().keys()
    # odict_keys(['fc2.bias', 'fc2.weight_orig', 'fc2.weight_mask', 'fc3.bias', 'fc3.weight_orig', 'fc3.weight_mask'])


def demo_global_pruning():

    model = LeNet()

    # if you want to manually specify target layers
    # parameters_to_prune = (
    #     (model.conv1, 'weight'),
    #     (model.conv2, 'weight'),
    #     (model.fc1, 'weight'),
    #     (model.fc2, 'weight'),
    #     (model.fc3, 'weight'),
    # )

    # list of tuple(layer, weight_name)
    parameters_to_prune = [
        (module, "weight") for module in list(model.modules())[1:]
    ]

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )

    print(
        "Sparsity in conv1.weight: {:.2f}%".format(
            100.0
            * float(torch.sum(model.conv1.weight == 0))
            / float(model.conv1.weight.nelement())
        )
    )
    print(
        "Sparsity in conv2.weight: {:.2f}%".format(
            100.0
            * float(torch.sum(model.conv2.weight == 0))
            / float(model.conv2.weight.nelement())
        )
    )
    print(
        "Sparsity in fc1.weight: {:.2f}%".format(
            100.0
            * float(torch.sum(model.fc1.weight == 0))
            / float(model.fc1.weight.nelement())
        )
    )
    print(
        "Sparsity in fc2.weight: {:.2f}%".format(
            100.0
            * float(torch.sum(model.fc2.weight == 0))
            / float(model.fc2.weight.nelement())
        )
    )
    print(
        "Sparsity in fc3.weight: {:.2f}%".format(
            100.0
            * float(torch.sum(model.fc3.weight == 0))
            / float(model.fc3.weight.nelement())
        )
    )
    print(
        "Global sparsity: {:.2f}%".format(
            100.0
            * float(
                torch.sum(model.conv1.weight == 0)
                + torch.sum(model.conv2.weight == 0)
                + torch.sum(model.fc1.weight == 0)
                + torch.sum(model.fc2.weight == 0)
                + torch.sum(model.fc3.weight == 0)
            )
            / float(
                model.conv1.weight.nelement()
                + model.conv2.weight.nelement()
                + model.fc1.weight.nelement()
                + model.fc2.weight.nelement()
                + model.fc3.weight.nelement()
            )
        )
    )


def main():

    # demo_layer_pruning()
    # demo_global_pruning()
    # demo_prune_remove()
    # demo_ln_structured_pruning()
    # demo_l1_unstructured_pruning()
    # demo_random_unstructured_pruning()
    demo_random_structured_pruning()


if __name__ == "__main__":
    main()
