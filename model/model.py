from attention_module import cbam_block
from time import time
from lookahead import Lookahead
import matplotlib.pyplot as plt
import os
import cv2
import keras
import numpy as np
from SSIM import DSSIMObjective
from sklearn.model_selection import train_test_split

# 设计GPU按需分
import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

lr = 0.001
epoch = 50
img_size = 128
img_channel = 3
batch_size = 2
k = 0.1
weight_path = 'D:/DeepLearning/Pycharm/PycharmProjects/UnderWater-Image-Enhancement/model/weight/MSCANet(0).h5'
data_augmentation = False


# 加载数据
def load_data(data_dir):
    # 读取文件中的图片
    image_list = os.listdir(data_dir)
    images = []  # 存放图片的列表
    for image in image_list:
        fd = os.path.join(data_dir, image)
        img = cv2.imread(fd, 1)  # 1-表示三通道，彩色图，0-表示单通道，灰度图
        data = cv2.resize(img, (img_size, img_size))
        images.append(data)

    x = np.array(images)/255
    x = x.reshape(x.shape[0], img_size, img_size, img_channel)
    x = x.astype('float32')
    print(np.shape(x))
    return x


# SC Module
def sc_block(input_map):
    block_conv1 = keras.layers.Conv2D(64, (3, 3), padding='same')(input_map)
    block_bn1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(block_conv1)
    block_Lrelu1 = keras.layers.LeakyReLU(alpha=0.05)(block_bn1)
    block_concat1 = keras.layers.concatenate([input_map, block_Lrelu1], axis=-1)
    block_conv2 = keras.layers.Conv2D(64, (3, 3), padding='same')(block_concat1)
    block_sc = cbam_block(block_conv2)
    block_concat2 = keras.layers.concatenate([input_map, block_sc], axis=-1)
    return block_concat2


# 模型
def model(input_shape):
    main_input = keras.Input(shape=input_shape, name='main_input')

    # 1X1卷积
    conv_1X1 = keras.layers.Conv2D(64, (1, 1), padding='same')(main_input)
    conv_1X1 = keras.layers.LeakyReLU(alpha=0.05)(conv_1X1)
    conv_1X1_block1 = sc_block(conv_1X1)
    conv_1X1_block2 = sc_block(conv_1X1_block1)
    conv_1X1_block3 = sc_block(conv_1X1_block2)
    conv_1X1_block4 = sc_block(conv_1X1_block3)
    conv_1X1_block10 = sc_block(conv_1X1_block4)
    conv_1X1_concat = keras.layers.concatenate([conv_1X1, conv_1X1_block10], axis=-1)
    conv_1X1_1 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_1X1_concat)
    conv_1X1_1 = keras.layers.LeakyReLU(alpha=0.05)(conv_1X1_1)

    # 3X3卷积
    conv_3X3 = keras.layers.Conv2D(64, (3, 3), padding='same')(main_input)
    conv_3X3 = keras.layers.LeakyReLU(alpha=0.05)(conv_3X3)
    conv_3X3_block1 = sc_block(conv_3X3)
    conv_3X3_block2 = sc_block(conv_3X3_block1)
    conv_3X3_block3 = sc_block(conv_3X3_block2)
    conv_3X3_block4 = sc_block(conv_3X3_block3)
    conv_3X3_block10 = sc_block(conv_3X3_block4)
    conv_3X3_concat = keras.layers.concatenate([conv_3X3, conv_3X3_block10], axis=-1)
    conv_3X3_1 = keras.layers.Conv2D(64, (3, 3), padding='same')(conv_3X3_concat)
    conv_3X3_1 = keras.layers.LeakyReLU(0.05)(conv_3X3_1)

    # 5X5卷积
    conv_5X5 = keras.layers.Conv2D(64, (5, 5), padding='same')(main_input)
    conv_5X5 = keras.layers.LeakyReLU(alpha=0.05)(conv_5X5)
    conv_5X5_block1 = sc_block(conv_5X5)
    conv_5X5_block2 = sc_block(conv_5X5_block1)
    conv_5X5_block3 = sc_block(conv_5X5_block2)
    conv_5X5_block4 = sc_block(conv_5X5_block3)
    conv_5X5_block10 = sc_block(conv_5X5_block4)
    conv_5X5_concat = keras.layers.concatenate([conv_5X5, conv_5X5_block10], axis=-1)
    conv_5X5_1 = keras.layers.Conv2D(64, (5, 5), padding='same')(conv_5X5_concat)
    conv_5X5_1 = keras.layers.LeakyReLU(alpha=0.05)(conv_5X5_1)

    # 融合
    fusion_concat = keras.layers.concatenate([conv_1X1_1, conv_3X3_1, conv_5X5_1], axis=-1)
    conv1 = keras.layers.Conv2D(64, (3, 3), padding='same')(fusion_concat)
    conv1 = keras.layers.LeakyReLU(alpha=0.05)(conv1)
    dp_block = keras.layers.SeparableConv2D(64, (3, 3), padding='same')(conv1)
    dp_block = keras.layers.LeakyReLU(alpha=0.05)(dp_block)
    cdp_block = keras.layers.multiply([conv1, dp_block])

    concat = keras.layers.concatenate([main_input, cdp_block], axis=-1)
    mse_output = keras.layers.Conv2D(img_channel, (1, 1), padding='same', name='mse_output')(concat)
    mse_output = keras.layers.LeakyReLU(alpha=0.05)(mse_output)

    model = keras.Model(inputs=main_input, outputs=mse_output)
    return model


# 拉普拉斯损失
def lap_loss(y_true, y_pred):
    print(np.shape(y_true))
    print(np.shape(y_pred))
    sobel_x = tf.constant([[1, 1, 1], [1, -8, 1], [1, 1, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [1, 3, 3, 1])
    sobel_true = tf.nn.conv2d(y_true, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    sobel_pred = tf.nn.conv2d(y_pred, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    sobel_loss = tf.reduce_mean(tf.square(sobel_true - sobel_pred))
    return sobel_loss


# 损失函数
def total_loss(y_true, y_pred):
    # 1.拉普拉斯损失
    lapla_loss = lap_loss(y_true, y_pred)

    # 2.MSE损失
    mse_loss = tf.reduce_mean(tf.square(y_true-y_pred))

    # 3.SSIM损失
    ssim = DSSIMObjective()
    ssim.__int__(y_pred)
    ssim_loss = ssim.__call__(y_true, y_pred)

    # 总的损失
    total_loss = mse_loss+ssim_loss+k*lapla_loss
    return total_loss


# 学习率衰减策略
def schedule(epoch):
    # 每隔10个epoch，学习率减小为原来的0.1
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    print("epoch-----lr", epoch, K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)


# 对数据进行分批
def data_generator(train_x, train_y, batch_size):
    batches = (len(train_x)+batch_size-1)//batch_size
    while(True):
        for i in range(batches):
            X = train_x[i*batch_size:(i+1)*batch_size]
            Y = train_y[i*batch_size:(i+1)*batch_size]
            yield (X, Y)


# 训练
def train(model, data):
    x, y = data
    # 将数据划分为训练集合验证集
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, shuffle=True)
    # 设置检查点
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=weight_path,
            monitor='loss',
            save_best_only=True
        ),
        keras.callbacks.LearningRateScheduler(schedule),
    ]
    # 编译模型
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss=total_loss)
    # 初始化Lookahead
    lookahead = Lookahead(k=5, alpha=0.5)
    # 插入到模型中
    lookahead.inject(model)
    # 开始时间
    start_time = time()
    # 训练模型
    if not data_augmentation:
        print("不使用数据分割")
        history = model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size,
                            validation_data=(valid_x, valid_y),
                            verbose=1, callbacks=callbacks_list)
    else:
        print("使用数据分割")
        history = model.fit_generator(generator=data_generator(train_x, train_y, batch_size),
                                      steps_per_epoch=(len(train_x)+batch_size-1)//batch_size,
                                      epochs=epoch, verbose=1, callbacks=callbacks_list,
                                      validation_data=data_generator(valid_x, valid_y, batch_size),
                                      validation_steps=(len(valid_x)+batch_size-1)//batch_size
                                      )
    model.save(weight_path)  # 保存模型

    # 结束时间
    duration = time()-start_time
    print("Train Finished takes:", "{:.2f} h".format(duration/3600.0))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()
    return model


# 测试
def test(model, data):
    # 解包数据
    test_x = data.reshape(1, img_size, img_size, img_channel)
    test_y = model.predict(test_x)
    recon = test_y.reshape(img_size, img_size, img_channel)
    recon = (np.array(recon))*255
    print(np.shape(recon))
    cv2.imwrite('D:/DeepLearning/Pycharm/PycharmProjects/UnderWater-Image-Enhancement/recon/test(0).jpg', recon)


if __name__ == '__main__':
    train_x = load_data('D:/DeepLearning/Pycharm/PycharmProjects/UnderWater-Image-Enhancement/data/'
                        'train_set/train')
    train_y = load_data('D:/DeepLearning/Pycharm/PycharmProjects/UnderWater-Image-Enhancement/data/'
                        'train_set/label')
    test_x = load_data('D:/DeepLearning/Pycharm/PycharmProjects/UnderWater-Image-Enhancement/data/'
                       'test_set/testA/test-50')
    model = model(input_shape=train_x.shape[1:])
    model.summary()
    model = train(model=model, data=(train_x, train_y))
    test(model=model, data=(test_x[1]))