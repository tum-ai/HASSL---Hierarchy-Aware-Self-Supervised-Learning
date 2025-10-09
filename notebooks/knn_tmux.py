
# grab directory root
import sys
sys.path.append("../")

from dinov3.eval.tSNE import extract_embeddings
from dinov3.data.datasets import NCells
from dinov3.models.backbone_loader import load_backbone
from dinov3.eval.simpleKNN import evaluate_simple_knn
import torch
import json
from torchvision import transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# defaul transform used in dinov3
def make_transform(resize_size: int | list[int] = 768):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])

test_dataset = NCells(
    root="/home/students/code/helmholtzSS25/Hannes/dinov3playground/manifest_test_fixed.csv.gz",
    split=NCells.Split.TRAIN,
    transform=make_transform(), 
    target_transform=None,
)
print(f"Test Dataset contains {len(test_dataset)} entries")


model = load_backbone(
    config_file="/home/students/code/helmholtzSS25/Hannes/dinov3playground/HDBSCAN_no_LAMBDA_WEIGHTING/config.yaml",
    pretrained_weights="/home/students/code/helmholtzSS25/Hannes/dinov3playground/HDBSCAN_no_LAMBDA_WEIGHTING/ckpt/22499",
    output_dir="/home/students/code/helmholtzSS25/Hannes/dinov3playground/HDBSCAN_no_LAMBDA_WEIGHTING/"
)
target_size = 224
batch_size = 64
num_workers = 6
embeddings, labels = extract_embeddings(
    model=model,
    dataset=test_dataset,
    device=DEVICE,
    batch_size=batch_size,
    num_workers=num_workers,
    target_size=target_size,
)

results = evaluate_simple_knn(embeddings, labels, k_list=[1,3,5,9], metric='cosine', sample_size=None)
print("kNN Performance:", json.dumps(results, sort_keys=True, indent=4))

results = evaluate_simple_knn(embeddings, labels, k_list=[1,3,5,9], metric='euclidean', sample_size=None)
print("kNN Performance:", json.dumps(results, sort_keys=True, indent=4))