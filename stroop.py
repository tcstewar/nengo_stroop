import nengo
import nengo_spa as spa
import numpy as np

words = ['CAR', 'CAT', 'BLUE', 'RED']
colors = ['RED', 'BLUE']

D = 32
vocab = spa.Vocabulary(D)
vocab.populate(';'.join(words))
vocab.populate('COLOR; WORD')

stimuli = []
for i in range(10):
    w = np.random.choice(words)
    c = np.random.choice(colors)
    stimuli.append((w,c))

model = spa.Network()
with model:
    
    t_stim = 0.5
    t_isi = 0.5
    def word_func(t):
        index = int (t / (t_stim + t_isi))
        t = t % (t_stim + t_isi)
        if t < t_isi:
            return '0'
        else:
            return stimuli[index%len(stimuli)][0]
    def color_func(t):
        index = int (t / (t_stim + t_isi))
        t = t % (t_stim + t_isi)
        if t < t_isi:
            return '0'
        else:
            return stimuli[index%len(stimuli)][1]
    stim_w = spa.Transcode(word_func, output_vocab=vocab)
    stim_c = spa.Transcode(color_func, output_vocab=vocab)    
    
    
    attention = spa.State(vocab)
    
    wm = spa.State(vocab)
    
    (spa.sym.COLOR*stim_c+spa.sym.WORD*stim_w)*~attention >> wm
    
    