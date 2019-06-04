import torch
from torch.autograd import Variable
from utils.wrap_var import to_var

use_cuda = torch.cuda.is_available()

def get_newq_lengths(new_questions, EOS,max_num=30):

    new_question_lengths = list()
    for q_idx in range(new_questions.size(0)):
        if EOS not in list(new_questions[q_idx].data):
            new_question_lengths.append(max_num)
            # print('Catching a corner case.') #Happened to me only once. Please report to @aashish if you encounter this
        else:
            new_question_lengths.append(list(new_questions[q_idx].data).index(EOS)+1)

    return to_var(torch.LongTensor(new_question_lengths))

def anspred2wordtok(answer_predictions, word2i):
    """
    Given the predictions over the classes No / Yes / N/A, will return the word tokens for the
    respective answer
    :param answer_predictions: Bx3
    :param word2i: dictionary, mapping words to word tokens
    :returns: Bx1
    """
    anspredIDX2anstok = {
        0: word2i['<no>'],
        1: word2i['<yes>'],
        2: word2i['<n/a>']
        }

    _tokens = answer_predictions.topk(1)[1].data
    answer_tokens = to_var(torch.LongTensor(_tokens.size(0), _tokens.size(1)).fill_(0))

    for j, i in enumerate(_tokens):
        answer_tokens[j, 0] = anspredIDX2anstok[i[0]]

    return answer_tokens

def append_dialogue(dialogue, dialogue_length, new_questions, question_length, answer_tokens, pad_token):
    """
    Given a dialogue history, will append a new question and its answer to it.
    Will take care of padding, possible cutting off (if required) the dialogue as well as
    returning updated dialogue length.
    :param dialogue: [Bx100]
    :param dialogue_length: [B]
    :param new_questions: [Bx15]
    :param question_length: [B]
    :param answer_tokens: [B,1]
    :param pad_token: int
    :returns: dialogue: [Bx100], dialogue_length: [B]
    """

    max_dialogue_length = dialogue.size(1)

    for qi, q in enumerate(new_questions):
        # put new dialogue together from old dialogue + new question + answer
        updated_dialogue = torch.cat(
            [
                dialogue[qi][:dialogue_length[qi].data[0]],
                new_questions[qi, :question_length.data[qi]],
                answer_tokens[qi]
            ]
        )

        # update length
        dialogue_length[qi] = min(updated_dialogue.size(0), max_dialogue_length)

        # strip and pad
        updated_dialogue = updated_dialogue[:max_dialogue_length]
        if updated_dialogue.size(0) < dialogue.size(1):
            dialogue_pad = to_var(torch.Tensor(max_dialogue_length - updated_dialogue.size(0)).fill_(pad_token).long())
            updated_dialogue = torch.cat([updated_dialogue, dialogue_pad])

        dialogue[qi] = updated_dialogue.unsqueeze(0)

    return dialogue, dialogue_length

def dialtok2dial(dialogue, i2word):
    """
    given a dialoage tensor (BxL) with word token ids, returns the words
    """
    if isinstance(dialogue, Variable):
        dialogue = dialogue.data

    batch_dial = list()

    for bid in range(dialogue.size(0)):
        dial = str()
        for i in dialogue[bid]:
            dial += i2word[str(i)] + ' '
            if dial.split()[-2:] == ["<padding>", "<padding>"]:
                dial = ' '.join(dial.split()[:-2])
                break

        batch_dial.append(dial)

    return batch_dial
