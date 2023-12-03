from pathlib import Path
import tempfile
import numpy as np
import tensorflow as tf
import transformers
from .. import common
from .dataset import ExtractSubtitlesText
from ..subtitles import Extract

max_length = 128

def to_features(texts, max_length):
    shape = (len(texts), max_length)
    # input_idsやattention_mask, token_type_idsの説明はglossaryに記載(cf. https://huggingface.co/transformers/glossary.html)
    input_ids = np.zeros(shape, dtype="int32")
    attention_mask = np.zeros(shape, dtype="int32")
    token_type_ids = np.zeros(shape, dtype="int32")

    tokenizer = transformers.BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    for i, text in enumerate(texts):
        encoded_dict = tokenizer.encode_plus(text, max_length=max_length, pad_to_max_length=True)
        input_ids[i] = encoded_dict["input_ids"]
        attention_mask[i] = encoded_dict["attention_mask"]
        token_type_ids[i] = encoded_dict["token_type_ids"]
    return [input_ids, attention_mask, token_type_ids]

class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path) -> None:
        model = tf.keras.models.load_model(r'speech.keras', custom_objects={'TFBertModel': transformers.TFBertModel})
        clips = self.Clips()
        with tempfile.TemporaryDirectory(prefix='speech') as tmpFolder:
            subtitlesPathList = Extract(videoPath, Path(tmpFolder))
            assPath = None
            for path in subtitlesPathList:
                if path.suffix == '.ass':
                    assPath = path
                    break
            textList = [ ExtractSubtitlesText(assPath, clip) for clip in clips ] if assPath is not None else [ '' ] * len(clips)
        X = to_features(textList, max_length)
        Y = model.predict(X)
        for i in range(len(clips)):
            self.Mark(clips[i], 'speech', float(Y[i][0]))
        self.Save()