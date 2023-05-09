print("Setting up")
from utlis import *
from sklearn.model_selection import train_test_split

path = 'myData'
data = importDataInfo(path)

balanceData(data, False)

imagespath, steerings = loadData(path,data)
# print(imagespath[0],steering[0])

X_train, X_valid, y_train, y_valid = train_test_split( imagespath, steerings, test_size=0.2, random_state=5)
print("Training Samples: {}\nValid Samples: {}".format(len(X_train), len(X_valid)))

model = createModel()
model.summary()

history = model.fit_generator(batchGen(X_train, y_train, 100, 1),steps_per_epoch=300, epochs=10,
validation_data=batchGen(X_valid, y_valid, 100, 0),validation_steps=200,verbose=1,shuffle = 1)

model.save('model.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')