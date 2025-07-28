from typing import Dict, Any
from lime.lime_text import LimeTextExplainer
import numpy as np
from .inference import predict_proba, get_class_names

_CLASS_NAMES = get_class_names()
_EXPLAINER = LimeTextExplainer(class_names=_CLASS_NAMES)

def explain_text(text: str, num_features: int = 10, num_samples: int = 500, target_class: int = None) -> Dict[str, Any]:
    def predict_fn(batch_texts):
        return predict_proba(batch_texts)

    # Always specify labels to avoid TypeError
    if target_class is not None:
        labels = [target_class]
    else:
        probs = predict_fn([text])[0]
        target_class = int(np.argmax(probs))
        labels = [target_class]

    exp = _EXPLAINER.explain_instance(
        text_instance=text,
        classifier_fn=predict_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=labels  # always a list
    )

    weights = exp.as_list(label=target_class)
    return {
        "text": text,
        "target_class_id": target_class,
        "target_class_name": _CLASS_NAMES[target_class],
        "weights": [{"term": w, "weight": float(v)} for w, v in weights]
    }
