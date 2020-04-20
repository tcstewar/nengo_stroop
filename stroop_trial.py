import nengo
import nengo_spa as spa
import numpy as np
import pytry

class StroopTrial(pytry.PlotTrial):
    def params(self):
        self.param('number of dimensions in semantic pointer', D=16)
        self.param('number of each type of stimuli to test', n_stims=2)
        self.param('inter-stimulus interval', t_isi=0.05)
        self.param('stimulus presentation time', t_stim=0.15)
        self.param('direct automaticity', auto_direct=0.0)
        self.param('use neurons', use_neurons=True)
        self.param('decision making system (bg|ia)', decision='bg')
        self.param('threshold for producing an output', output_threshold=0.1)
        self.param('IA decision time', ia_accum_timescale=0.4)
        self.param('attention error', attention_error=0.0)
        
        
    def evaluate(self, p, plt):
        stimuli = []
        for i in range(p.n_stims):
            stimuli.append(('NEUTRAL%d'%i, 'COLOR%d'%i, 'neutral'))
        for i in range(p.n_stims):
            stimuli.append(('COLOR%d'%i, 'COLOR%d'%i, 'congruent'))
        for i in range(p.n_stims):
            stimuli.append(('COLOR%d'%((i+1)%p.n_stims), 'COLOR%d'%i, 'incongruent'))

        vocab = spa.Vocabulary(p.D, pointer_gen=np.random.RandomState(seed=p.seed))
        for i in range(p.n_stims):
            vocab.populate('NEUTRAL%d'%i)
            vocab.populate('COLOR%d'%i)
        vocab.populate('COLOR; WORD')

        model = spa.Network(seed=p.seed)
        with model:
            def word_func(t):
                index = int (t / (p.t_stim + p.t_isi))
                t = t % (p.t_stim + p.t_isi)
                if t < p.t_isi:
                    return '0'
                else:
                    return stimuli[index%len(stimuli)][0]
            def color_func(t):
                index = int (t / (p.t_stim + p.t_isi))
                t = t % (p.t_stim + p.t_isi)
                if t < p.t_isi:
                    return '0'
                else:
                    return stimuli[index%len(stimuli)][1]
            stim_w = spa.Transcode(word_func, output_vocab=vocab)
            stim_c = spa.Transcode(color_func, output_vocab=vocab) 
            stim_a = spa.Transcode('(1-%g)*COLOR + %g*WORD' % (p.attention_error, p.attention_error), output_vocab=vocab)

            wm = spa.State(vocab)
            
            (spa.sym.COLOR*stim_c+spa.sym.WORD*stim_w)*~stim_a >> wm
            
            if p.auto_direct != 0:
                stim_w*p.auto_direct >> wm
                
                
            speech = spa.State(vocab)
            
            if p.decision == 'bg':
                with spa.ActionSelection() as action_sel:
                    for i in range(p.n_stims):
                        spa.ifmax(spa.dot(wm, spa.sym('COLOR%d'%i)), spa.sym('COLOR%d'%i) >> speech)                
                    spa.ifmax(0.35, spa.sym('0') >> speech)
            elif p.decision == 'ia':
                def reset_func(t):
                    index = int (t / (p.t_stim + p.t_isi))
                    t = t % (p.t_stim + p.t_isi)
                    if t < p.t_isi:
                        return 1
                    else:
                        return 0                                
                reset = nengo.Node(reset_func)
                
                decision = spa.IAAssocMem(vocab, mapping=['COLOR%d'%i for i in range(p.n_stims)],
                                          accum_timescale=p.ia_accum_timescale)
                wm >> decision
                
                nengo.Connection(reset, decision.input_reset)
            else:
                raise Exception('Unknown decision param: %s' %p.decision)
            if not p.use_neurons:
                for ens in model.all_ensembles:
                    ens.neuron_type=nengo.Direct()
                

            p_output = nengo.Probe(wm.output, synapse=0.02)
            p_correct = nengo.Probe(stim_c.output)
            p_speech = nengo.Probe(speech.output, synapse=0.02)
            
            if p.decision == 'bg':
                p_act = nengo.Probe(action_sel.thalamus.output, synapse=0.01)
            elif p.decision == 'ia':
                p_act = nengo.Probe(decision.selection.output, synapse=0.01)
            

        sim = nengo.Simulator(model, progress_bar=p.verbose)
        with sim:
            sim.run(p.n_stims*(p.t_isi+p.t_stim)*3)    

        v = np.einsum('ij,ij->i',sim.data[p_correct], sim.data[p_output])
        steps = int((p.t_isi+p.t_stim)/sim.dt)
        scores = v[steps-2::steps]
        

        data = sim.data[p_act]
        rts = []
        accuracy = []
        for condition in range(3):
            for i in range(p.n_stims):
                t_start = (p.t_isi+p.t_stim)*i + p.t_isi + condition*(p.t_isi+p.t_stim)*p.n_stims
                t_end = t_start + p.t_stim
                d = data[int(t_start/sim.dt):int(t_end/sim.dt) , i]

                correct = np.max(d) > p.output_threshold
                if correct:
                    rt = np.where(d > p.output_threshold)[0][0]*sim.dt
                else:
                    rt = None

                rts.append(rt)
                accuracy.append(correct)
        
        
        
        
        
        if plt:
            plt.subplot(2, 1, 1)
            plt.plot(sim.trange(), sim.data[p_output].dot(vocab.vectors.T))
            plt.subplot(2, 1, 2)
            if p.decision == 'bg':
                plt.plot(sim.trange(), sim.data[p_act][:,:-1])
            elif p.decision == 'ia':
                plt.plot(sim.trange(), sim.data[p_act])
                
            for i in range(p.n_stims*3):
                plt.axvline(i*(p.t_isi+p.t_stim)+p.t_isi, ls='--')
            
        
        return dict(
            scores=scores,
            stimuli=stimuli,
            neutral=np.mean(scores[:p.n_stims]),
            congruent=np.mean(scores[p.n_stims:p.n_stims*2]),
            incongruent=np.mean(scores[p.n_stims*2:]),
            rts=rts,
            accuracy=accuracy,
            rt_neutral=np.mean(rts[:p.n_stims]),
            rt_congruent=np.mean(rts[p.n_stims:p.n_stims*2]),
            rt_incongruent=np.mean(rts[p.n_stims*2:]),
        )

        