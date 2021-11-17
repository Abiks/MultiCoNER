from transformers import MarianMTModel, MarianTokenizer
from easynmt import EasyNMT

def parse_dataset(filename):
    sentences = []
    tokens = []
    labels = []
    id = ''
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            # if line refers to id
            line_t = line.replace('\n', '')
            if len(line_t) == 0:
                if len(tokens) > 0:
                    sentences.append((id, tokens, labels))
            # if line is empty
            elif line_t[0] == '#':
                id = line_t[5:line_t.find('domain') - 1]
                tokens = []
                labels = []
            # if line refers to token and label
            else:
                token = line_t[:line_t.find('_') - 1]
                label = line_t[line_t.rfind('_') + 2:]
                tokens.append(token)
                labels.append(label)
    return sentences

def translate_sentence(model, sent, tokenizer):
    translated = model.generate(**tokenizer(sent, return_tensors="pt", padding=True))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

def easynmt():
    file_prefix = 'sample_data-'
    file_suffix = '.conll'
    sents = parse_dataset(file_prefix + 'tr' + file_suffix)
    sentences = [' '.join(x[1]) for x in sents]
    model = EasyNMT('mbart50_m2m')
    for sent in sentences:
        translation = model.translate(sent, target_lang='en')
        print(sent + '\t' + translation)


def marianmt():
    file_prefix = 'sample_data-'
    file_suffix = '.conll'
    sents = parse_dataset(file_prefix + 'tr' + file_suffix)
    sentences = [' '.join(x[1]) for x in sents]
    model_name = 'Helsinki-NLP/opus-mt-tr-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    for sent in sentences:
        translation = translate_sentence(model, sent, tokenizer)
        print(sent + '\t' + translation[0])

if __name__ == '__main__':
    easynmt()