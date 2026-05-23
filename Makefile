GLUE_TASKS := cola sst2 mrpc stsb qqp mnli qnli rte wnli

RESULTS_DIR ?= results/glue
SUBMISSIONS_DIR ?= results/submissions

BERT_EMBEDDING ?= bert-base-uncased
CHARACTERBERT_EMBEDDING ?= general_character_bert

EPOCHS ?= 3
TRAIN_BATCH_SIZE ?= 32
EVAL_BATCH_SIZE ?= 32
LEARNING_RATE ?= 2e-5
MAX_SEQ_LENGTH ?= 128
SEED ?= 42

MAX_TRAIN_EXAMPLES ?=
MAX_VALIDATION_EXAMPLES ?=
MAX_TEST_EXAMPLES ?=

UV := uv run --extra finetuning
FINETUNE := $(UV) character-bert-finetune
BERT_SUBMISSION_ZIP = $(SUBMISSIONS_DIR)/bert-base-uncased-glue.zip
CHARACTERBERT_SUBMISSION_ZIP = $(SUBMISSIONS_DIR)/general-character-bert-glue.zip
GLUE_SUBMISSION_FILES := CoLA.tsv SST-2.tsv MRPC.tsv STS-B.tsv QQP.tsv MNLI-m.tsv MNLI-mm.tsv AX.tsv QNLI.tsv RTE.tsv WNLI.tsv

LIMIT_ARGS = $(if $(MAX_TRAIN_EXAMPLES),--max-train-examples $(MAX_TRAIN_EXAMPLES),)
LIMIT_ARGS += $(if $(MAX_VALIDATION_EXAMPLES),--max-validation-examples $(MAX_VALIDATION_EXAMPLES),)
LIMIT_ARGS += $(if $(MAX_TEST_EXAMPLES),--max-test-examples $(MAX_TEST_EXAMPLES),)

COMMON_ARGS = --num-train-epochs $(EPOCHS)
COMMON_ARGS += --train-batch-size $(TRAIN_BATCH_SIZE)
COMMON_ARGS += --eval-batch-size $(EVAL_BATCH_SIZE)
COMMON_ARGS += --learning-rate $(LEARNING_RATE)
COMMON_ARGS += --max-seq-length $(MAX_SEQ_LENGTH)
COMMON_ARGS += --seed $(SEED)
COMMON_ARGS += --do-train
COMMON_ARGS += --do-predict
COMMON_ARGS += --write-glue-submission
COMMON_ARGS += $(LIMIT_ARGS)

.PHONY: help
help:
	@echo "GLUE fine-tuning targets:"
	@echo "  make finetune-bert-<task>            Run one GLUE task with bert-base-uncased"
	@echo "  make finetune-characterbert-<task>   Run one GLUE task with CharacterBERT"
	@echo "  make finetune-bert-glue              Run every GLUE task with bert-base-uncased"
	@echo "  make finetune-characterbert-glue     Run every GLUE task with CharacterBERT"
	@echo "  make glue-submission-bert            Run every BERT GLUE task and zip one submission"
	@echo "  make glue-submission-characterbert   Run every CharacterBERT GLUE task and zip one submission"
	@echo "  make glue-submissions                Run both per-model submission commands"
	@echo "  make smoke-glue-submission-bert      Tiny BERT training, full test TSVs, one zip"
	@echo "  make smoke-glue-submission-characterbert"
	@echo "                                      Tiny CharacterBERT training, full test TSVs, one zip"
	@echo "  make smoke-glue-submissions          Run both smoke submission commands"
	@echo ""
	@echo "Supported tasks: $(GLUE_TASKS)"
	@echo ""
	@echo "Useful overrides:"
	@echo "  EPOCHS=3 TRAIN_BATCH_SIZE=32 EVAL_BATCH_SIZE=32 LEARNING_RATE=2e-5 MAX_SEQ_LENGTH=128"
	@echo "  MAX_TRAIN_EXAMPLES=128 MAX_VALIDATION_EXAMPLES=64 MAX_TEST_EXAMPLES=64"

define GLUE_TARGET
.PHONY: finetune-$(1)-$(2)
finetune-$(1)-$(2):
	$(FINETUNE) \
		--dataset $(2) \
		--embedding $($(3)) \
		--output-dir $(RESULTS_DIR)/$($(3))/$(2) \
		$(COMMON_ARGS)
endef

$(foreach task,$(GLUE_TASKS),$(eval $(call GLUE_TARGET,bert,$(task),BERT_EMBEDDING)))
$(foreach task,$(GLUE_TASKS),$(eval $(call GLUE_TARGET,characterbert,$(task),CHARACTERBERT_EMBEDDING)))

.PHONY: finetune-bert-glue
finetune-bert-glue: $(addprefix finetune-bert-,$(GLUE_TASKS))

.PHONY: finetune-characterbert-glue
finetune-characterbert-glue: $(addprefix finetune-characterbert-,$(GLUE_TASKS))

.PHONY: finetune-glue-all
finetune-glue-all: finetune-bert-glue finetune-characterbert-glue

.PHONY: glue-submission-bert
glue-submission-bert: finetune-bert-glue
	mkdir -p $(SUBMISSIONS_DIR)/bert-base-uncased-glue
	rm -f $(BERT_SUBMISSION_ZIP)
	cp $(RESULTS_DIR)/$(BERT_EMBEDDING)/cola/glue_submission/CoLA.tsv $(SUBMISSIONS_DIR)/bert-base-uncased-glue/CoLA.tsv
	cp $(RESULTS_DIR)/$(BERT_EMBEDDING)/sst2/glue_submission/SST-2.tsv $(SUBMISSIONS_DIR)/bert-base-uncased-glue/SST-2.tsv
	cp $(RESULTS_DIR)/$(BERT_EMBEDDING)/mrpc/glue_submission/MRPC.tsv $(SUBMISSIONS_DIR)/bert-base-uncased-glue/MRPC.tsv
	cp $(RESULTS_DIR)/$(BERT_EMBEDDING)/stsb/glue_submission/STS-B.tsv $(SUBMISSIONS_DIR)/bert-base-uncased-glue/STS-B.tsv
	cp $(RESULTS_DIR)/$(BERT_EMBEDDING)/qqp/glue_submission/QQP.tsv $(SUBMISSIONS_DIR)/bert-base-uncased-glue/QQP.tsv
	cp $(RESULTS_DIR)/$(BERT_EMBEDDING)/mnli/glue_submission/MNLI-m.tsv $(SUBMISSIONS_DIR)/bert-base-uncased-glue/MNLI-m.tsv
	cp $(RESULTS_DIR)/$(BERT_EMBEDDING)/mnli/glue_submission/MNLI-mm.tsv $(SUBMISSIONS_DIR)/bert-base-uncased-glue/MNLI-mm.tsv
	cp $(RESULTS_DIR)/$(BERT_EMBEDDING)/mnli/glue_submission/AX.tsv $(SUBMISSIONS_DIR)/bert-base-uncased-glue/AX.tsv
	cp $(RESULTS_DIR)/$(BERT_EMBEDDING)/qnli/glue_submission/QNLI.tsv $(SUBMISSIONS_DIR)/bert-base-uncased-glue/QNLI.tsv
	cp $(RESULTS_DIR)/$(BERT_EMBEDDING)/rte/glue_submission/RTE.tsv $(SUBMISSIONS_DIR)/bert-base-uncased-glue/RTE.tsv
	cp $(RESULTS_DIR)/$(BERT_EMBEDDING)/wnli/glue_submission/WNLI.tsv $(SUBMISSIONS_DIR)/bert-base-uncased-glue/WNLI.tsv
	cd $(SUBMISSIONS_DIR)/bert-base-uncased-glue && python -m zipfile -c ../bert-base-uncased-glue.zip $(GLUE_SUBMISSION_FILES)

.PHONY: glue-submission-characterbert
glue-submission-characterbert: finetune-characterbert-glue
	mkdir -p $(SUBMISSIONS_DIR)/general-character-bert-glue
	rm -f $(CHARACTERBERT_SUBMISSION_ZIP)
	cp $(RESULTS_DIR)/$(CHARACTERBERT_EMBEDDING)/cola/glue_submission/CoLA.tsv $(SUBMISSIONS_DIR)/general-character-bert-glue/CoLA.tsv
	cp $(RESULTS_DIR)/$(CHARACTERBERT_EMBEDDING)/sst2/glue_submission/SST-2.tsv $(SUBMISSIONS_DIR)/general-character-bert-glue/SST-2.tsv
	cp $(RESULTS_DIR)/$(CHARACTERBERT_EMBEDDING)/mrpc/glue_submission/MRPC.tsv $(SUBMISSIONS_DIR)/general-character-bert-glue/MRPC.tsv
	cp $(RESULTS_DIR)/$(CHARACTERBERT_EMBEDDING)/stsb/glue_submission/STS-B.tsv $(SUBMISSIONS_DIR)/general-character-bert-glue/STS-B.tsv
	cp $(RESULTS_DIR)/$(CHARACTERBERT_EMBEDDING)/qqp/glue_submission/QQP.tsv $(SUBMISSIONS_DIR)/general-character-bert-glue/QQP.tsv
	cp $(RESULTS_DIR)/$(CHARACTERBERT_EMBEDDING)/mnli/glue_submission/MNLI-m.tsv $(SUBMISSIONS_DIR)/general-character-bert-glue/MNLI-m.tsv
	cp $(RESULTS_DIR)/$(CHARACTERBERT_EMBEDDING)/mnli/glue_submission/MNLI-mm.tsv $(SUBMISSIONS_DIR)/general-character-bert-glue/MNLI-mm.tsv
	cp $(RESULTS_DIR)/$(CHARACTERBERT_EMBEDDING)/mnli/glue_submission/AX.tsv $(SUBMISSIONS_DIR)/general-character-bert-glue/AX.tsv
	cp $(RESULTS_DIR)/$(CHARACTERBERT_EMBEDDING)/qnli/glue_submission/QNLI.tsv $(SUBMISSIONS_DIR)/general-character-bert-glue/QNLI.tsv
	cp $(RESULTS_DIR)/$(CHARACTERBERT_EMBEDDING)/rte/glue_submission/RTE.tsv $(SUBMISSIONS_DIR)/general-character-bert-glue/RTE.tsv
	cp $(RESULTS_DIR)/$(CHARACTERBERT_EMBEDDING)/wnli/glue_submission/WNLI.tsv $(SUBMISSIONS_DIR)/general-character-bert-glue/WNLI.tsv
	cd $(SUBMISSIONS_DIR)/general-character-bert-glue && python -m zipfile -c ../general-character-bert-glue.zip $(GLUE_SUBMISSION_FILES)

.PHONY: glue-submissions
glue-submissions: glue-submission-bert glue-submission-characterbert

.PHONY: smoke-glue-submission-bert
smoke-glue-submission-bert:
	$(MAKE) glue-submission-bert \
		RESULTS_DIR=results/smoke/glue \
		SUBMISSIONS_DIR=results/smoke/submissions \
		EPOCHS=1 \
		TRAIN_BATCH_SIZE=2 \
		EVAL_BATCH_SIZE=2 \
		MAX_SEQ_LENGTH=64 \
		MAX_TRAIN_EXAMPLES=8 \
		MAX_VALIDATION_EXAMPLES=4 \
		MAX_TEST_EXAMPLES=

.PHONY: smoke-glue-submission-characterbert
smoke-glue-submission-characterbert:
	$(MAKE) glue-submission-characterbert \
		RESULTS_DIR=results/smoke/glue \
		SUBMISSIONS_DIR=results/smoke/submissions \
		EPOCHS=1 \
		TRAIN_BATCH_SIZE=2 \
		EVAL_BATCH_SIZE=2 \
		MAX_SEQ_LENGTH=64 \
		MAX_TRAIN_EXAMPLES=8 \
		MAX_VALIDATION_EXAMPLES=4 \
		MAX_TEST_EXAMPLES=

.PHONY: smoke-glue-submissions
smoke-glue-submissions: smoke-glue-submission-bert smoke-glue-submission-characterbert
