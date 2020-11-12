# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import cv2
# from PIL import Image
# from glob import glob

# female = glob('freeai/data/cropFemale/*.png')
# male = glob('freeai/data/cropMale/*.png')

# path = male + female
# # getting size of image


# def getSize(path):
#     img = Image.open(path)
#     return img.size[0]


# s = getSize(path[0])
# print(s)

# # create data frame
# df = pd.DataFrame(data=path, columns=['path'])
# # print(df.head())
# # print(df.shape)

# df['size'] = df['path'].apply(getSize)
# # print(df.head())

# # print(df.tail())

# # print(df.describe())

# # plt.hist(df['size'], bins=25)
# # plt.show()

# # so resize all images in 100*100 form nd remve image with size less or equal to 54
# df_new = df[df['size'] > 60]
# # print(df_new)

# string = df_new['path'][0]
# print(string.split('_')[0].split('/'))


# def Gender(string):
#     try:
#         string.split('_')[0][-1]
#     except:
#         return None


# df['Gender'] = df['path'].apply(Gender)
# print(df)
# df['Gender'].value_counts(normalize=True)

# # 60 % are female and rest are male
# # all the img size is mre tan or equal to 81

# df['path'][0]
# path_to_resize = df['path'][0]
# print(path_to_resize)


# def resize_img(path_to_resize):
#     try:
#         # step1 read image
#         img = cv2.imread(path_to_resize)
#         # step2 convert into grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         # step3 resize into 100*100 array
#         size = gray.shape[0]
#         if size >= 100:
#             gray_re = cv2.resize(gray, (100, 100), cv2.INTER_AREA)  # SHRINK

#         else:
#             gray_re = cv2.resize(gray, (100, 100), cv2.INTER_CUBIC)  # ENLARGE
#         # step 4 flatten the image(1*10000)
#         return gray_re.flatten()
#     except:
#         return None


# df_new['Gender'] = df_new['path'].apply(Gender)
# df_new.head()

# # structre data
# df_new['structure_data'] = df_new['path'].apply(resize_img)

# print(df)


female = glob('freeai/data/cropFemale/*.png')
male = glob('freeai/data/cropMale/*.png')

path = female + male


def getSize(path):
    img = Image.open(path)
    return img.size[0]


df = pd.DataFrame(data=path, columns=['path'])
df.head()  # display top 5 rows

df['size'] = df['path'].apply(getSize)
df.head()
df.tail()
df.describe()

plt.hist(df['size'], bins=30)
plt.show()

df_new = df[df['size'] > 60]

string = df_new['path'][0]


def gender(string):
    try:

        return string.split('_')[0].split('/')[-1]
    except:
        return None


df['gender'] = df['path'].apply(gender)

print(df['gender'].value_counts(normalize=True))
df['gender'].value_counts(normalize=True).plot(kind='bar')
plt.show()


def resize_img(path_to_resize):
    try:

        # step - 1: read image
        img = cv2.imread(path_to_resize)
        # step - 2: convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # step -3: resize into 100 x 100 array
        size = gray.shape[0]
        if size >= 100:  # shrink
            gray_re = cv2.resize(gray, (100, 100), cv2.INTER_AREA)  # SHRINK
        else:  # enlarge
            gray_re = cv2.resize(gray, (100, 100), cv2.INTER_CUBIC)  # ENLARGE
        # step -4: Flatten Image (1x10,000)
        return gray_re.flatten()
    except:
        return None


len(resize_img(path[0]))

df_new['gender'] = df_new['path'].apply(gender)
df_new.head()

# structuring function
df_new['structure_data'] = df_new['path'].apply(resize_img)

# copy and expand their columns
df1 = df_new['structure_data'].apply(pd.Series)
df2 = pd.concat((df_new['gender'], df1), axis=1)
df2.head()

plt.imshow(df2.loc[0][1:].values.reshape(100, 100).astype('int'), cmap='gray')
plt.title("Label: "+df2.loc[0]['gender'])
plt.show()

pickle.dump(df2, open('freeai/data/dataframe_images_100_100.pickle', 'wb'))
print(df)
