

class Example(object):
    """A single training/test example for the roc dataset."""
    def __init__(self,
                 input_id,
                 obs,
                 hyps,
                 ans = None,
                 adjacancy = None,
                 ask_for = None
                 ):
        self.input_id = input_id
        self.obs = obs
        self.hyps = hyps
        self.ans = ans - 1
        self.adjacancy = adjacancy
        #self.ask_for = ask_for
       

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 answer,
                 # hyp_id,
                 obs_sentences,
                 hyps,
                 compared_features,
                 compared_answer

    ):
        self.example_id = example_id
        try:
            self.choices_features = [
                {
                    'tokens': tokens,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'sentence_ind': sentence_ind,
                    'graph': graph,
                    'sentence_ids':graph_embedding
                }
                for tokens, input_ids, input_mask, sentence_ind, graph, graph_embedding in choices_features
            ]
        except:
            self.choices_features = [
                {
                    'tokens': tokens,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'sentence_ind': sentence_ind,
                    'graph': graph,
                    'hyp_ids': hyp_ids_tmp,
                    'hyp_mask': hyp_mask_tmp
                }
                for tokens, input_ids, input_mask, sentence_ind, graph, hyp_ids_tmp, hyp_mask_tmp in choices_features
            ]
        self.answer = answer

        self.compared_feature =  [
                {
                    'com_tokens': tokens,
                    'com_input_ids': input_ids,
                    'com_input_mask': input_mask,
                    'com_sentence_ind': sentence_ind,
                }
                for tokens, input_ids, input_mask, sentence_ind in compared_features
            ]

        self.compared_answer = compared_answer

        # self.hyp_id = hyp_id
        self.obs_sentences = obs_sentences
        self.hyps=hyps