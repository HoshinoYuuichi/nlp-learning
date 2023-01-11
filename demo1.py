import jieba
jieba.setLogLevel(jieba.logging.INFO)
from jieba import lcut_for_search
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
#定义文本
texts = ["坚定不移地走中国特色社会主义道路",
         "矢志不渝地坚定以经济发展为中心",
         "科学技术是第一生产力",
         "创新是第一驱动力"]
#读取文本并通过jieba.lcut_for_search进行分词
texts = [lcut_for_search(text) for text in texts]
#建立词典，收集稀疏向量
dictionary = Dictionary(texts)
num_features = len(dictionary.token2id)
#建立语料库
corpus = [dictionary.doc2bow(text) for text in texts]
#输入查询文本
search_word = input("请输入：")
#收集查询文本的稀疏向量
sw_vec = dictionary.doc2bow(lcut_for_search(search_word))
#对定义文本进行基于tfidf的词袋表示，得到tfidf模型
tfidf = TfidfModel(corpus)
tf_texts = tfidf[corpus]
#将查询文本通过前面定义的词典转换为词袋模型
tf_sw = tfidf[sw_vec]
#构建矩阵运算，计算文本相似度
sparse_matrix = SparseMatrixSimlarity(tf_texts, num_features)
similarities = sparse_matrix.get_similarities(tf_sw)
#list设置元组
list = []
for e, s in enumerate(similarities, 1):
  print("搜索内容与文本%d的相似度为：%.2f" % (e, s))
  a = (e, s)
  list.append(a)
#元组列表排序
list = sorted(list, key = lambda x:x[1])
#输出最大值
print("其中相似度最大的为文本%d, 相似度为：%.2f" % (list[3][0], list[3][1]))
