#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import torch
from torch import nn

from data_handling.annotated_sentence import AnnotatedSentence
from data_handling.dependency_matrix import DependencyMatrix
from data_handling.tag_sequence import TagSequence
from models.outputs import DependencyClassifier, SequenceTagger, ArcScorer


class MultiParser(nn.Module):
    """This is the main module of the parsing system, tying together input and output(s). It operates by first
    retrieving input embeddings for each task via the underlying transformer-based language model, and then
    feeding these embeddings to the respective output modules. The outputs are then combined into AnnotatedSentence
    objects.
    """

    def __init__(self, embeddings_processor, outputs, post_processors=None):
        """
        Args:
            embeddings_processor: Module to produce embeddings for the output tasks (e.g. BERTWrapper).
            outputs: ModuleDict of modules that produce the actual parser outputs (e.g. DependencyClassifier).
            post_processors: List of components which post-process module output (e.g. FactorizedMSTPostProcessor;
              default: None).
        """
        super(MultiParser, self).__init__()

        # Source of token embeddings
        self.embed = embeddings_processor

        # ModuleDict of modules that produce the actual parser outputs
        self.outputs = outputs

        # Annotation types and label vocabs for the outputs
        self.annotation_types = self._get_annotation_types()
        self.label_vocabs = {outp_id: self.outputs[outp_id].vocab for outp_id in self.outputs}

        # List of components which post-process module output (optional)
        self.post_processors = post_processors if post_processors is not None else list()
        
        self.lang_embeds = nn.Embedding(5, 100) #change: 5 LID vectors for the 5 LID categories in the SAGT Treebank. The number should be changed for other treebanks when the number of LID categories is different. 

    def parse(self, sentence):
        """Parse a singular sentence (in evaluation mode, i.e. no dropout) and perform post-processing.

        Args:
            sentence: The sentence to be parsed. If sentence is of type str, input is assumed to be a
              whitespace-tokenized "raw" sentence. If sentence is of type DependencyAnnotatedSentence, tokenization will
              be taken from that sentence.

        Returns:
            An AnnotatedSentence instance with the predicted relations.
        """
        # Extract sentence tokens and make dummy batch
        if isinstance(sentence, AnnotatedSentence):
            tokens = sentence.tokens[1:]  # Omit [root] token
            lids = sentence.lids[1:] #change: getting LIDs from the sentence.
            multiword_tokens = sentence.multiword_tokens
        elif isinstance(sentence, str):
            tokens = sentence.split(" ")
            multiword_tokens = None
        else:
            raise Exception("Sentence must be either whitespace-tokenized string or DependencyAnnotatedSentence!")
        singleton_batch = [tokens]
        lid_batch = [lids] #change: getting LID batch

        # Ensure eval mode and compute logits, labels
        self.eval()
        logits, labels = self._compute_logits_and_labels(singleton_batch, lid_batch) #change: sending LID batch to compute_logits_and_labels

        # Squeeze to get rid of dummy batch dimension
        logits = {outp_id: torch.squeeze(logits[outp_id], dim=0) for outp_id in logits}
        labels = {outp_id: torch.squeeze(labels[outp_id], dim=0) for outp_id in labels}

        # Create AnnotatedSentence from label tensors
        parsed_sentence = AnnotatedSentence.from_tensors(["[root]"] + tokens, lids, labels, self.label_vocabs,
                                                         self.annotation_types, multiword_tokens=multiword_tokens) #change: added lids to AnnotatedSentence

        # Post-process those annotation layers for which post-processing modules have been provided
        #change: removed the post_processor that is needed for dependency parsing
        #for post_processor in self.post_processors:
        #    post_processor.post_process(parsed_sentence, logits)

        return parsed_sentence

    def evaluate_batch(self, gold_sentences, post_process=False):
        """Run the parser on a batch of gold AnnotatedSentences and compute parsing metrics w.r.t. to the provided
        gold annotations. Optionally, run sentence post-processing.

        Args:
            gold_sentences: List of gold AnnotatedSentences to run the parser on.
            post_process: If True, post-processing will be performed on the parsed sentences. Default: False.

        Returns:
            The raw model output (logits) as well as a dictionary containing the evaluation counts for each
            annotation layer. (For the time being, these evaluation counts are for "TOTAL" only, i.e. we don't
            care about the counts for the individual labels.)
        """
        # Create token batch and compute logits
        sent_batch = [sent.tokens_no_root() for sent in gold_sentences]
        lid_batch = [sent.lids[1:] for sent in gold_sentences] #change: getting the LID batch for each sentence batch
        logits, labels = self._compute_logits_and_labels(sent_batch, lid_batch) #change: sending LID batch to compute_logits_and_labels

        # Iterate over the sentences in the batch and compute eval metrics for each
        batch_metrics = {outp_id: {"predicted": 0, "gold": 0, "correct": 0} for outp_id in self.outputs}
        for i, gold_sentence in enumerate(gold_sentences):
            # Create a predicted AnnotatedSentence for each input sentence
            curr_logits = {outp_id: logits[outp_id][i] for outp_id in logits}
            curr_labels = {outp_id: labels[outp_id][i] for outp_id in labels}

            predicted_sentence = AnnotatedSentence.from_tensors(gold_sentence.tokens, gold_sentence.lids, curr_labels, self.label_vocabs,
                                                                self.annotation_types) #change: added lids to AnnotatedSentence
            if post_process:
                for outp_id in self.post_processors:
                    self.post_processors.post_process(predicted_sentence[outp_id], curr_logits[outp_id])

            # Evaluate sentence against gold annotations and update batch counts.
            # We only care about the total counts here (might change this at a later point)
            instance_metrics = AnnotatedSentence.get_annotation_counts(gold_sentence, predicted_sentence)
            instance_metrics = {outp_id: instance_metrics[outp_id]["TOTAL"] for outp_id in instance_metrics}
            update_eval_counts(batch_metrics, instance_metrics)

        return logits, batch_metrics

    def _compute_logits_and_labels(self, input_sents, input_lids): #change: added input LIDs
        """For the given batch of sentences (provided as a list of lists of tokens), compute logits and labels
        for each output/annotation ID by first running the embeddings processor and then the individual output modules.
        The output modules also handle the conversion from logits to labels (argmax).
        """
        # Get token embeddings
        embeddings, true_seq_lengths = self.embed(input_sents, input_lids) #change: added input LIDs
        
        
        #change: from line 141 to line 158: concatenation of LID vectors to the word embeddings.
        word_to_ix = {"TR": 0, "DE": 1, "MIXED": 2, "LANG3": 3, "OTHER": 4} #This is for the SAGT Treebank. Need to change it for other treebanks.
      
        embeddings_new = dict()
        for output_id in self.outputs:
            embeddings_new[output_id] = torch.zeros(list(embeddings[output_id].size())[0],list(embeddings[output_id].size())[1],868).cuda()
         
        for i,sent in enumerate(input_sents):
            for j,word in enumerate(sent):
                for output_id in self.outputs:
                    
                    lookup_tensor = torch.tensor([word_to_ix[input_lids[i][j]]], dtype=torch.long)
                    lid_embed = self.lang_embeds(lookup_tensor.cuda())
                    if embeddings[output_id].size()[2] == 768:
                       embeddings_new[output_id][i][j] = torch.cat((embeddings[output_id][i][j],  lid_embed[0].cuda()),dim=0)
                    else:
                       embeddings_new[output_id][i][j] = torch.cat((embeddings[output_id][i][j][:-100],  lid_embed[0].cuda()),dim=0)
                    
        bn = nn.BatchNorm1d(868).cuda() #change: batch normalization for the concatenated embeddings.

        # Run the output modules on the embeddings
        logits = dict()
        labels = dict()
        for output_id in self.outputs:
           
            #change: batch normalization for the concatenated embeddings.
            permuted = embeddings_new[output_id].permute(0, 2, 1)
            output = bn(permuted)
            embeddings_new[output_id] = output.permute(0,2,1)

            curr_logits, curr_labels = self.outputs[output_id](embeddings_new[output_id], true_seq_lengths) #change: replaced embeddings with embeddings_new
            logits[output_id] = curr_logits
            labels[output_id] = curr_labels

        return logits, labels

    def _get_annotation_types(self):
        """Get the kinds of annotations that this MultiParser produces. Returns a dict: Output ID -> Annotation type."""
        annotation_types = dict()
        for outp_id in self.outputs:
            if type(self.outputs[outp_id]) is DependencyClassifier:
                annotation_types[outp_id] = DependencyMatrix
            elif type(self.outputs[outp_id]) is SequenceTagger:
                annotation_types[outp_id] = TagSequence
            elif type(self.outputs[outp_id]) is ArcScorer:
                if self.outputs[outp_id].head_mode == "single_head":
                    annotation_types[outp_id] = TagSequence
                elif self.outputs[outp_id].head_mode == "multi_head":
                    annotation_types[outp_id] = DependencyMatrix
                else:
                    raise Exception("ArcScorer has unknown head mode!")
            else:
                raise Exception("Unknown output module {}".format(type(self.outputs[outp_id])))

        return annotation_types

    def parallelize(self, device_ids):
        """Distribute this parser over multiple devices. For now, this only affects the outputs, as they (presumably)
        require the most memory.

        Args:
            device_ids: List of device IDs to distribute the model over.
        """
        self.embed.parallelize(device_ids)
        for outp_id in self.outputs:
            self.outputs[outp_id] = torch.nn.DataParallel(self.outputs[outp_id], device_ids=device_ids)


def update_eval_counts(aggregate_metrics, curr_metrics):
    """For each output ID, update the counts in aggregate_metrics by adding the counts in curr_metrics."""
    for outp_id in curr_metrics:
        if outp_id not in aggregate_metrics:
            aggregate_metrics[outp_id] = {"predicted": 0, "gold": 0, "correct": 0}

        for count in curr_metrics[outp_id]:
            assert count in aggregate_metrics[outp_id]
            aggregate_metrics[outp_id][count] += curr_metrics[outp_id][count]
