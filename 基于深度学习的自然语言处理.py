#!/usr/bin/env python
# coding: utf-8

# # 数据集概览

# In[3]:


from datasets import load_dataset

# 加载 IMDB 数据集
dataset = load_dataset("imdb")

# 查看数据集基本结构
print("数据集信息:")
print(dataset)

# 检查训练集、测试集的键
print("\n训练集字段:", dataset['train'].column_names)
print("测试集字段:", dataset['test'].column_names)

# 查看数据集大小
print("\n数据集大小:")
print(f"训练集: {len(dataset['train'])} 条")
print(f"测试集: {len(dataset['test'])} 条")

# 查看训练集的前3条数据
print("\n训练集前3条数据:")
print(dataset['train'][:3])

# 查看单条数据的示例
print("\n单条训练数据示例:")
print(dataset['train'][0])


# # 数据加载和预处理

# In[5]:


import re
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 加载 IMDB 数据集
dataset = load_dataset("imdb")

# 定义文本清洗函数
def preprocess_text_basic(text):
    # 转换为小写
    text = text.lower()
    # 去除标点符号和非字母字符
    text = re.sub(r"[^a-z\s]", "", text)
    # 分词
    words = text.split()
    # 去除停用词
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    # 返回清洗后的文本
    return " ".join(words)

# 对训练集和测试集进行预处理
train_data = [preprocess_text_basic(text) for text in dataset['train']['text']]
test_data = [preprocess_text_basic(text) for text in dataset['test']['text']]

# 获取标签
train_labels = dataset['train']['label']
test_labels = dataset['test']['label']

# 打印预处理后的样例
print("清洗前的样例数据:", dataset['train'][:1])
print("清洗后的样例数据:", train_data[:1])


# # 数据分割

# In[6]:


from sklearn.model_selection import train_test_split

# 划分训练集、验证集
X_train, X_val, y_train, y_val = train_test_split(
    train_data, train_labels, test_size=0.2, random_state=42
)
X_test = test_data
y_test = test_labels


# # 向量化数据

# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
# 使用TF-IDF向量化
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 保存 TF-IDF 向量化器
tfidf_vectorizer_path =r"D:\新建文件夹\tfidf_vectorizer.pkl"
with open(tfidf_vectorizer_path, "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"TF-IDF 向量化器已保存到 {tfidf_vectorizer_path}")
# 打印样例向量
print("TF-IDF样例向量矩阵:", X_train_tfidf[0])


# In[34]:


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 转换为Tensor
X_train_tensor = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_tfidf.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_tfidf.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
batch_size =32
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)


# # 定义简单模型

# In[39]:


# 定义简单LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        _, (hn, _) = self.lstm(x.unsqueeze(1), (h0, c0))
        out = self.fc(hn[-1])
        return out



# # 更加复杂的模型

# In[35]:


import torch
from torch import nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout_prob=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 使用多层双向 LSTM
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,  # 双向 LSTM
            dropout=dropout_prob  # Dropout 用于正则化
        )
        
        # 全连接层输入维度乘以 2，正向和反向隐藏层拼接
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # 初始化隐藏状态和记忆单元（适配多层和双向）
        h0 = torch.zeros(2 * self.lstm.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(2 * self.lstm.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM 前向传播
        _, (hn, _) = self.lstm(x.unsqueeze(1), (h0, c0))
        
        # 取最后一个时间步的正向和反向隐藏状态拼接
        hn_cat = torch.cat((hn[-2], hn[-1]), dim=1)
        
        # Dropout 正则化
        hn_cat = self.dropout(hn_cat)
        
        # 全连接层用于分类
        out = self.fc(hn_cat)
        return out


# In[36]:


#input_dim = X_train_tfidf.shape[1]
#hidden_dim = 128
#output_dim = 2
# 定义模型参数
input_dim = X_train_tfidf.shape[1]
hidden_dim = 64
output_dim = 2
num_layers = 2
dropout_prob = 0.5
#model = LSTMClassifier(input_dim, hidden_dim, output_dim)
# 调用模型
model = LSTMClassifier(input_dim, hidden_dim, output_dim, num_layers, dropout_prob)
# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 训练过程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(10): 
    model.train()
    epoch_loss = 0
    correct = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()#梯度清零
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == y_batch).sum().item()
    accuracy = correct / len(X_train)
    print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={accuracy:.4f}")
# 保存模型
model_save_path = r"D:\新建文件夹\lstm_model.pth"  # 模型保存路径
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存到 {model_save_path}")


# # 模型评估

# In[37]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'SimHei'

# 在验证集上评估
model.eval()
with torch.no_grad():
    val_preds = []
    val_probs = []  # 存储正类概率，用于 AUC-ROC 计算
    val_labels = []
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)  # 转换为概率分布
        _, preds = torch.max(outputs, 1)
        val_preds.extend(preds.cpu().numpy())
        val_probs.extend(probs[:, 1].cpu().numpy())  # 正类的概率
        val_labels.extend(y_batch.cpu().numpy())

# 评估指标
print("分类报告:")
print(classification_report(val_labels, val_preds))

# 混淆矩阵
cm = confusion_matrix(val_labels, val_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("混淆矩阵")
plt.show()

# AUC-ROC 分数计算
auc = roc_auc_score(val_labels, val_probs)
print(f"AUC-ROC 分数: {auc:.4f}")

# 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')  # 对角线表示随机猜测
plt.xlabel("假正率 (False Positive Rate, FPR)")  # 横轴标签
plt.ylabel("真正率 (True Positive Rate, TPR)")  # 纵轴标签
plt.title("ROC 曲线 (Receiver Operating Characteristic Curve)")  # 标题
plt.legend(loc="best")  # 图例位置
plt.grid()  # 网格线
plt.show()



# # 模型测试

# In[40]:


import torch
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from torch import nn

# 定义简单的LSTM模型
'''
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        _, (hn, _) = self.lstm(x.unsqueeze(1), (h0, c0))
        out = self.fc(hn[-1])
        return out
        '''
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout_prob=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 使用多层双向 LSTM
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,  # 双向 LSTM
            dropout=dropout_prob  # Dropout 用于正则化
        )
        
        # 全连接层输入维度需要乘以 2，因为双向 LSTM 的输出是正向和反向隐藏层拼接
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # 初始化隐藏状态和记忆单元（适配多层和双向）
        h0 = torch.zeros(2 * self.lstm.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(2 * self.lstm.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM 前向传播
        _, (hn, _) = self.lstm(x.unsqueeze(1), (h0, c0))
        
        # 取最后一个时间步的正向和反向隐藏状态拼接
        hn_cat = torch.cat((hn[-2], hn[-1]), dim=1)
        
        # Dropout 正则化
        hn_cat = self.dropout(hn_cat)
        
        # 全连接层用于分类
        out = self.fc(hn_cat)
        return out

# 定义文本清洗函数
def preprocess_text_basic(text):
    # 转换为小写
    text = text.lower()
    # 去除标点符号和非字母字符
    text = re.sub(r"[^a-z\s]", "", text)
    # 分词
    words = text.split()
    # 去除停用词
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    # 返回清洗后的文本
    return " ".join(words)

# 加载保存的模型权重
'''device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 5000  # TF-IDF 的特征维度
hidden_dim = 128
output_dim = 2
model = LSTMClassifier(input_dim, hidden_dim, output_dim).to(device)'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 5000
hidden_dim = 64
output_dim = 2
num_layers = 2
dropout_prob = 0.5
#model = LSTMClassifier(input_dim, hidden_dim, output_dim)
# 调用模型
model = LSTMClassifier(input_dim, hidden_dim, output_dim, num_layers, dropout_prob)
# 假设你已经保存了模型权重
model_save_path =r"D:\新建文件夹\lstm_model.pth"  # 模型保存路径
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# 加载保存的 TF-IDF 向量化器
tfidf_vectorizer_path =r"D:\新建文件夹\tfidf_vectorizer.pkl"  # TF-IDF 向量化器路径
import pickle
with open(tfidf_vectorizer_path, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# 定义预测函数
def predict_review(review, model, vectorizer):
    # 数据预处理
    clean_review = preprocess_text_basic(review)
    # TF-IDF 向量化
    review_vector = vectorizer.transform([clean_review]).toarray()
    review_tensor = torch.tensor(review_vector, dtype=torch.float32).to(device)
    # 模型预测
    with torch.no_grad():
        outputs = model(review_tensor)
        _, predicted = torch.max(outputs, 1)
    # 映射结果到标签
    label_map = {0: "负面评论", 1: "正面评论"}
    return label_map[predicted.item()]

# 输入英文电影评论
while True:
    print("\n请输入电影评论（输入 'exit' 退出程序）:")
    news_comment = input().strip()
    if news_comment.lower() == "exit":
        print("退出程序！")
        break

    # 预测并输出结果
    result = predict_review(news_comment, model, tfidf_vectorizer)
    print(f"评论情感预测结果: {result}")


# In[31]:


import random
from sklearn.model_selection import ParameterSampler

# 定义超参数搜索空间
param_space = {
    "lr": [0.0001, 0.001, 0.01, 0.05],
    "batch_size": [16,32, 64, 128],
    "hidden_dim": [64, 128, 256],
    "epochs": [10, 15, 20]
}

# 随机采样超参数组合
n_trials = 50
param_combinations = list(ParameterSampler(param_space, n_iter=n_trials, random_state=42))
print("随机生成的超参数组合:")
for i, params in enumerate(param_combinations, 1):
    print(f"组合 {i}: {params}")

# 定义训练和验证函数
def train_and_evaluate(params):
    # 获取超参数
    lr = params["lr"]
    batch_size = params["batch_size"]
    hidden_dim = params["hidden_dim"]
    epochs = params["epochs"]

    # 数据加载器
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

    # 定义模型
    model = LSTMClassifier(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练过程
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
        accuracy = correct / len(X_train_tensor)

    # 验证过程
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == y_batch).sum().item()
    val_accuracy = val_correct / len(X_val_tensor)
    return val_accuracy

# 遍历超参数组合并记录最佳性能
best_params = None
best_accuracy = 0

for i, params in enumerate(param_combinations, 1):
    print(f"\n开始训练超参数组合 {i}: {params}")
    val_accuracy = train_and_evaluate(params)
    print(f"验证集准确率: {val_accuracy:.4f}")
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = params

print("\n最佳超参数组合:")
print(best_params)
print(f"最佳验证集准确率: {best_accuracy:.4f}")


# In[ ]:




