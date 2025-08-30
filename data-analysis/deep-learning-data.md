# 深度学习数据处理 Prompts

## 数据集准备专家

```
请为深度学习项目准备数据集：

【项目信息】
- 任务类型：[分类/回归/分割/检测/生成]
- 模型架构：[CNN/RNN/Transformer/GAN]
- 数据模态：[图像/文本/音频/多模态]
- 数据规模：[样本数量]
- 硬件限制：[GPU内存/存储空间]

【数据处理pipeline】

1. **数据加载与存储优化**
   ```python
   import tensorflow as tf
   import torch
   from torch.utils.data import Dataset, DataLoader
   
   # TFRecord格式（TensorFlow）
   def create_tfrecord(data_path, output_path):
       writer = tf.io.TFRecordWriter(output_path)
       for sample in data:
           feature = {
               'image': tf.train.Feature(
                   bytes_list=tf.train.BytesList(value=[image])),
               'label': tf.train.Feature(
                   int64_list=tf.train.Int64List(value=[label]))
           }
           example = tf.train.Example(
               features=tf.train.Features(feature=feature))
           writer.write(example.SerializeToString())
   
   # 自定义Dataset（PyTorch）
   class CustomDataset(Dataset):
       def __init__(self, data_path, transform=None):
           self.data_path = data_path
           self.transform = transform
           self.data = self.load_data()
       
       def __getitem__(self, idx):
           sample = self.data[idx]
           if self.transform:
               sample = self.transform(sample)
           return sample
   ```

2. **数据增强策略**
   ```python
   # 图像增强
   from albumentations import (
       Compose, RandomRotate90, Flip, Transpose,
       OneOf, MotionBlur, MedianBlur, GaussianBlur,
       ShiftScaleRotate, OpticalDistortion, GridDistortion,
       HueSaturationValue, RandomBrightnessContrast,
       CLAHE, RandomGamma, CoarseDropout, Cutout,
       MixUp, CutMix
   )
   
   augmentation = Compose([
       RandomRotate90(p=0.5),
       Flip(p=0.5),
       OneOf([
           MotionBlur(p=0.2),
           MedianBlur(blur_limit=3, p=0.1),
           GaussianBlur(p=0.2),
       ], p=0.5),
       ShiftScaleRotate(shift_limit=0.1, 
                        scale_limit=0.2, 
                        rotate_limit=30, p=0.5),
       OneOf([
           OpticalDistortion(p=0.3),
           GridDistortion(p=0.1),
       ], p=0.2),
       HueSaturationValue(p=0.3),
       RandomBrightnessContrast(p=0.3),
       CoarseDropout(max_holes=8, max_height=32, 
                    max_width=32, p=0.5)
   ])
   
   # 文本增强
   import nlpaug.augmenter.word as naw
   import nlpaug.augmenter.sentence as nas
   
   # 同义词替换
   aug = naw.SynonymAug(aug_src='wordnet')
   # 回译
   aug = naw.BackTranslationAug(
       from_model_name='Helsinki-NLP/opus-mt-en-de',
       to_model_name='Helsinki-NLP/opus-mt-de-en'
   )
   ```

3. **数据平衡处理**
   ```python
   from imblearn.over_sampling import SMOTE, ADASYN
   from imblearn.under_sampling import RandomUnderSampler
   from imblearn.combine import SMOTETomek
   
   # 过采样
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X, y)
   
   # 类权重调整
   from sklearn.utils.class_weight import compute_class_weight
   class_weights = compute_class_weight(
       'balanced', 
       classes=np.unique(y), 
       y=y
   )
   
   # Focal Loss for imbalanced data
   class FocalLoss(nn.Module):
       def __init__(self, alpha=1, gamma=2):
           super().__init__()
           self.alpha = alpha
           self.gamma = gamma
   ```

4. **数据验证与分割**
   ```python
   from sklearn.model_selection import (
       StratifiedKFold, 
       TimeSeriesSplit,
       GroupKFold
   )
   
   # 分层K折交叉验证
   skf = StratifiedKFold(n_splits=5, shuffle=True)
   
   # 时间序列分割
   tscv = TimeSeriesSplit(n_splits=5)
   
   # 组别分割（防止数据泄露）
   gkf = GroupKFold(n_splits=5)
   
   # 验证集策略
   def create_validation_split(df, val_size=0.2):
       # 确保标签分布一致
       # 考虑时间因素
       # 避免数据泄露
       pass
   ```

5. **批处理优化**
   ```python
   # 动态批处理
   from torch.nn.utils.rnn import pad_sequence
   
   def collate_fn(batch):
       # 处理变长序列
       sequences = [item['sequence'] for item in batch]
       labels = [item['label'] for item in batch]
       
       # 填充到相同长度
       padded = pad_sequence(sequences, batch_first=True)
       return padded, torch.tensor(labels)
   
   # 预取和缓存
   dataset = dataset.prefetch(tf.data.AUTOTUNE)
   dataset = dataset.cache()
   ```

6. **数据质量监控**
   ```python
   # 数据漂移检测
   from alibi_detect.cd import KSDrift, MMDDrift
   
   cd = KSDrift(X_ref, p_val=0.05)
   preds = cd.predict(X_test)
   
   # 标签噪声检测
   from cleanlab import find_label_issues
   
   issues = find_label_issues(
       labels=labels,
       pred_probs=pred_probs,
       return_indices_ranked_by='self_confidence'
   )
   ```
```

## 模型训练数据分析

```
请分析深度学习模型训练过程中的数据问题：

【训练日志】
```
Epoch 1/100: loss=2.31, acc=0.45, val_loss=2.28, val_acc=0.47
Epoch 2/100: loss=2.15, acc=0.52, val_loss=2.35, val_acc=0.46
...
```

【问题诊断】

1. **过拟合检测与解决**
   ```python
   # 早停策略
   from tensorflow.keras.callbacks import EarlyStopping
   
   early_stop = EarlyStopping(
       monitor='val_loss',
       patience=10,
       restore_best_weights=True
   )
   
   # Dropout增强
   class DropoutScheduler(Callback):
       def __init__(self, initial_drop=0.1, increment=0.1):
           self.drop_rate = initial_drop
           self.increment = increment
       
       def on_epoch_end(self, epoch, logs=None):
           if logs['val_loss'] > logs['loss'] * 1.1:
               # 增加dropout
               self.drop_rate = min(0.5, 
                                   self.drop_rate + self.increment)
   ```

2. **梯度问题分析**
   ```python
   # 梯度监控
   def gradient_monitor(model):
       gradients = []
       for name, param in model.named_parameters():
           if param.grad is not None:
               grad_norm = param.grad.data.norm(2).item()
               gradients.append({
                   'layer': name,
                   'grad_norm': grad_norm,
                   'vanishing': grad_norm < 1e-5,
                   'exploding': grad_norm > 100
               })
       return gradients
   
   # 梯度裁剪
   torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                  max_norm=1.0)
   ```

3. **学习率调度**
   ```python
   # 自适应学习率
   from torch.optim.lr_scheduler import (
       CosineAnnealingLR,
       OneCycleLR,
       ReduceLROnPlateau
   )
   
   # Warmup策略
   def warmup_lr_scheduler(optimizer, warmup_epochs):
       def lr_lambda(epoch):
           if epoch < warmup_epochs:
               return float(epoch) / float(max(1, warmup_epochs))
           return 1.0
       return LambdaLR(optimizer, lr_lambda)
   ```

4. **批归一化诊断**
   ```python
   # 监控BN层统计量
   def check_bn_stats(model):
       for module in model.modules():
           if isinstance(module, nn.BatchNorm2d):
               print(f"Mean: {module.running_mean.mean().item()}")
               print(f"Var: {module.running_var.mean().item()}")
               
               # 检查是否更新
               if module.training:
                   assert module.running_mean.requires_grad == False
   ```

5. **数据加载性能分析**
   ```python
   import time
   
   # 数据加载瓶颈分析
   def profile_dataloader(dataloader, num_batches=10):
       load_times = []
       process_times = []
       
       for i, batch in enumerate(dataloader):
           if i >= num_batches:
               break
           
           start = time.time()
           # 数据传输到GPU
           batch = batch.cuda()
           transfer_time = time.time() - start
           
           # 模型前向传播
           output = model(batch)
           process_time = time.time() - start - transfer_time
           
           load_times.append(transfer_time)
           process_times.append(process_time)
       
       print(f"Avg load time: {np.mean(load_times):.4f}s")
       print(f"Avg process time: {np.mean(process_times):.4f}s")
   ```
```