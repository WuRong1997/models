# 应用mfcc清洗脏数据
import librosa
path = '***.wav'
def wav2mfcc(path):
    y,sr = librosa.load(path=path, sr=None, mono=None)
    y = y[::3]
    flag = 0
    for elem in y:
        if abs(elem) > 0.00000001:
            flag = 1
            break
        if flag == 1: #音频有效
        if flag == 0: #音频无效

# 提取音频的特征表示：记录有效音频的id并下载有效音频，使用vggish提取音频的特征表示，在vggish_inference_demo.py文件中获取shape为(1,128)的特征表示
examples_batch = vggish_input.wavfile_to_examples(path)
[embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: examples_batch})
postprocessed_batch = pproc.postprocess(embedding_batch)
if postprocessed_batch.shape[0] == 1:
    postprocessed_batch
else:
    postprocessed_batch = tf.reduce_mean(postprocessed_batch, axis=0, keep_dims=True).eval()
np.save(postprocessed_batch)

# 使用特征表示训练分类模型

# 切割音频并分声道
import copy
from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np
path = '***.wav'
audio = AudioSegment.from_wav(path)
audio_copy = copy.deepcopy(audio)
start_time = start*1000
end_time = end*1000
audio_split = audio_copy[start_time:end_time]
save_file = '***.wav'
audio_split.export(save_file, format='wav')
sampleRate, musicData = wavfile.read(save_file)
left = []
right = []
for item in musicData:
    left.append(item[0])
    right.append(item[1])
left_file = '***.wav'
right_file = '***.wav'
wavfile.write(left_file, sampleRate, np.array(left))
wavfile.write(right_file, sampleRate, np.array(right))

