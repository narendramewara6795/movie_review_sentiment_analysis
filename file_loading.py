import pandas as pd
import glob
pd.set_option('display.max_columns', None)

trainPos = glob.glob("/home/sunbeam/Documents/aclImdb_v1/aclImdb/train/pos/*.txt")
trainNeg = glob.glob("/home/sunbeam/Documents/aclImdb_v1/aclImdb/train/neg/*.txt")
trainUnsup = glob.glob("/home/sunbeam/Documents/aclImdb_v1/aclImdb/train/unsup/*.txt")
testPos = glob.glob("/home/sunbeam/Documents/aclImdb_v1/aclImdb/test/pos/*.txt")
testNeg = glob.glob("/home/sunbeam/Documents/aclImdb_v1/aclImdb/test/neg/*.txt")

url_train_pos = pd.read_csv("/home/sunbeam/Documents/aclImdb_v1/aclImdb/train/urls_pos.txt",header=None)
url_train_neg = pd.read_csv("/home/sunbeam/Documents/aclImdb_v1/aclImdb/train/urls_neg.txt",header=None)
url_train_unsup = pd.read_csv("/home/sunbeam/Documents/aclImdb_v1/aclImdb/train/urls_unsup.txt",header=None)
url_test_pos = pd.read_csv("/home/sunbeam/Documents/aclImdb_v1/aclImdb/test/urls_pos.txt",header=None)
url_test_neg = pd.read_csv("/home/sunbeam/Documents/aclImdb_v1/aclImdb/test/urls_neg.txt",header=None)

def get_movieIds(urls):
    movieIds = []
    for index in range(len(urls)):
        movieIds.append(urls[index].split("/")[4])
    return movieIds

def function(param,urls):
    files = []
    reviews = []
    id = []
    ratings = []
    sentiments = []
    for index in range(len(param)):
        url = param[index]
        text = open(url, "r")
        data = text.read()
        reviews.append(data)
        fileName = param[index].split("/")[8]
        files.append(fileName)
        fileNames = fileName.split("_")
        id.append(int(fileNames[0]))
        rating = int(fileNames[1].split(".")[0])
        ratings.append(rating)
        if rating >= 7:
            sentiments.append("pos")
        elif rating <= 4:
            sentiments.append("neg")
        else:
            sentiments.append("neu")

    movieIds = get_movieIds(urls.values[:,0])

    df = pd.concat([pd.DataFrame({"index":id,"rating":ratings,"sentiment":sentiments,"review":reviews,"file":files})])
    df = df.sort_values(by="index")

    df["movieId"] = movieIds
    val = ["movieId","rating","sentiment","review","file"]
    df = df[val]
    # print(df)
    return df


df1 = function(trainPos,url_train_pos)
df2 = function(trainNeg,url_train_neg)
df3 = function(testPos,url_test_pos)
df4 = function(testNeg,url_test_neg)

train = pd.concat([df1,df2])
train.to_csv(f"./dataset/train.csv", index=False)
test = df = pd.concat([df3,df4])
test.to_csv(f"./dataset/test.csv", index=False)

movie_dataset = pd.concat([df1,df2,df3,df4])
movie_dataset.to_csv(f"./dataset/movie_dataset.csv", index=False)
