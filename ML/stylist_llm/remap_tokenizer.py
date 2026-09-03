"""
remap_tokenizer.py -- wraps the ORIGINAL (unpruned) tokenizer plus
prune_vocabulary.py's token_id_map.json to give the pruned model a
consistent encode()/decode() interface. See prune_vocabulary.py's module
docstring for why the tokenizer itself is never rewritten -- only the
model's embedding/lm_head weight matrices are pruned, and ids are remapped
at this boundary instead.

An id the original tokenizer produces that is NOT in the retained set
raises RuntimeError immediately, rather than silently indexing into the
wrong embedding row -- this should never happen if prune_vocabulary.py's
floor (tag_vocabulary.full_vocabulary_terms()) actually covers everything
the training/inference prompts can contain; if it does happen, that's a
real coverage bug to fix in tag_vocabulary.py, not something to paper over
here with an UNK fallback that would quietly degrade generation quality.
"""
import json
from pathlib import Path


class RemappedTokenizer:
    def __init__(self, base_tokenizer, token_id_map_path):
        self.base = base_tokenizer
        with open(token_id_map_path) as f:
            mapping = json.load(f)
        self.old_to_new = {int(k): v for k, v in mapping["old_to_new"].items()}
        self.new_to_old = mapping["new_to_old"]  # list, index == new_id
        self.vocab_size = len(self.new_to_old)

    @classmethod
    def from_pruned_dir(cls, vocab_dir, base_model_id=None):
        from transformers import AutoTokenizer
        vocab_dir = Path(vocab_dir)
        with open(vocab_dir / "token_id_map.json") as f:
            resolved_base_id = base_model_id or json.load(f)["base_model_id"]
        base_tokenizer = AutoTokenizer.from_pretrained(resolved_base_id)
        return cls(base_tokenizer, vocab_dir / "token_id_map.json")

    def encode(self, text, add_special_tokens=False):
        old_ids = self.base.encode(text, add_special_tokens=add_special_tokens)
        new_ids = []
        for old_id in old_ids:
            if old_id not in self.old_to_new:
                raise RuntimeError(
                    f"Token id {old_id} ('{self.base.decode([old_id])}') from input text is not in the "
                    f"pruned vocabulary. This means tag_vocabulary.full_vocabulary_terms() is missing a "
                    f"term used in this prompt -- fix the floor in tag_vocabulary.py and re-run "
                    f"prune_vocabulary.py, don't silently drop or remap this token."
                )
            new_ids.append(self.old_to_new[old_id])
        return new_ids

    def decode(self, new_ids, skip_special_tokens=True):
        old_ids = [self.new_to_old[i] for i in new_ids]
        return self.base.decode(old_ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages, add_generation_prompt=True):
        """Uses the base tokenizer's chat template (ChatML) to build the
        prompt STRING, then encodes through the remap -- never builds ids
        via the base tokenizer's own chat-template id output directly,
        since those ids are in the ORIGINAL vocabulary space."""
        text = self.base.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return self.encode(text, add_special_tokens=False)
