# KCOC：糖尿病视网膜病变有序分类

本仓库对应论文 *Imbalanced Ordinal Classification for Diabetic Retinopathy Grading with k-Positive Contrastive Learning*。统一入口是
`ordinal_classification/train_kcoc.py`。发布版本已移除绑定作者服务器的历史实验脚本、重复副本及临时分析文件。

## 方法与代码对应

KCOC 分为两个阶段：

1. `pretrain`：使用动量编码器、动态字典队列和每个样本固定 `k` 个同类正样本学习表征。
2. `finetune`：加载第一阶段的 query encoder，冻结编码器，只训练线性分类器，并使用 OCE 损失。

OCE 实现与论文一致：

```text
d(y, y') = 2 |y-y'| / C
adjusted_logits = logits + beta * d(y, y')
```

其中 `beta=0` 等价于普通交叉熵，论文最终实验使用 `beta=4`。

## 安装

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 数据清单

三个数据集统一使用 CSV 清单，必须包含 `path,label` 两列。`path` 可以是相对 CSV 所在目录的路径，也可以是绝对路径，标签必须为 0–4。

```csv
path,label
images/0001.png,0
images/0002.png,2
```

- APTOS2019、Messidor-2：准备五折的 `train_fold0.csv` 与 `test_fold0.csv`，一直到 fold 4。
- DDR：按官方划分准备 `train.csv`、`valid.csv`、`test.csv`。

可使用 `ordinal_classification/make_folds.py` 从一个总清单生成可复现的分层五折划分：

```bash
python ordinal_classification/make_folds.py --manifest all.csv --output splits/aptos --folds 5 --seed 591884092
```

## 眼底图像预处理

论文使用视野裁剪、CLAHE 和缩放。整理后的脚本会同时产生处理后的图片及新清单：

```bash
python ordinal_classification/preprocess_fundus.py --manifest all.csv --output data/aptos_clahe --size 512
```

## 训练

以下参数对应论文的默认设置：ResNet-50、128 维投影、队列 1024、`k=64`、动量 0.999、温度 0.07、第一阶段 200 epoch、第二阶段 30 epoch。

```bash
python ordinal_classification/train_kcoc.py pretrain \
  --train-csv splits/aptos/train_fold0.csv --output outputs/aptos/fold0 \
  --backbone resnet50 --image-size 224 --epochs 200 --batch-size 128 \
  --lr 0.016 --queue-size 1024 --num-positive 64 --momentum 0.999 --temperature 0.07 \
  --balanced-sampling

python ordinal_classification/train_kcoc.py finetune \
  --train-csv splits/aptos/train_fold0.csv --val-csv splits/aptos/test_fold0.csv \
  --pretrained outputs/aptos/fold0/pretrain_best.pth --output outputs/aptos/fold0 \
  --backbone resnet50 --image-size 224 --epochs 30 --batch-size 128 --lr 0.1 --beta 4 \
  --balanced-sampling

python ordinal_classification/train_kcoc.py evaluate \
  --data-csv splits/aptos/test_fold0.csv \
  --checkpoint outputs/aptos/fold0/finetune_best.pth --backbone resnet50 --image-size 224
```

512×512 输入通常需要按显存降低 batch size。DDR 可将 `--backbone` 改为 `densenet121`。三个子命令都支持 `--device cpu`，便于做小规模验证。

## 输出与复现说明

训练目录保存 checkpoint 和 JSON 指标。评估输出 Accuracy、balanced accuracy、quadratic weighted Kappa、macro/weighted F1、macro recall、macro precision 和 multiclass AUC。

数据集本身、五折清单和论文训练权重不包含在当前仓库中，因此在没有原始数据及权重的情况下，无法验证论文表格中的具体数值；但整理后的入口可以完整执行论文描述的两阶段流程。

论文 PDF 位于 [`paper/KCOC_paper.pdf`](paper/KCOC_paper.pdf)。预训练权重的发布说明见 [`MODEL_WEIGHTS.md`](MODEL_WEIGHTS.md)。

## License

本项目采用 [MIT License](LICENSE)。仓库中包含的第三方代码仍保留其原始版权声明。

## Citation

如果本项目对你的研究有帮助，请引用仓库中的论文；GitHub 也会根据 [CITATION.cff](CITATION.cff) 提供引用信息。
