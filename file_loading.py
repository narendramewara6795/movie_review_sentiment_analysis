import pandas as pd
import glob

trainPos = glob.glob("./dataset/aclImdb/train/pos/*.txt")
trainNeg = glob.glob("./dataset/aclImdb/train/neg/*.txt")
testPos = glob.glob("./dataset/aclImdb/test/pos/*.txt")
testNeg = glob.glob("./dataset/aclImdb/test/neg/*.txt")

url_train_pos = pd.read_csv("./dataset/aclImdb/train/urls_pos.txt",header=None)
url_train_neg = pd.read_csv("./dataset/aclImdb/train/urls_neg.txt",header=None)
url_test_pos = pd.read_csv("./dataset/aclImdb/test/urls_pos.txt",header=None)
url_test_neg = pd.read_csv("./dataset/aclImdb/test/urls_neg.txt",header=None)

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
        fileName = param[index].split("/")[5]
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
