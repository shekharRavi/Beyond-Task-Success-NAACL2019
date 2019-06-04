from tensorboardX import SummaryWriter

class Visualise():
    """docstring for Visualise."""
    # TODO: Add documentation for visualisation
    def __init__(self, **kwargs):
        super(Visualise, self).__init__()
        self.vis_args = kwargs
        self.writer = SummaryWriter(self.vis_args['tb_logdir']+self.vis_args['model_name']+'_'+self.vis_args['exp_name']+'_'+self.vis_args['ts'])

        self.train_id = 0 # train count for visualisation
        self.valid_id = 0 # valid count for visualisation
        if 'modulo' in kwargs:
            if kwargs['modulo']:
                self.modulo_train_id = 0
                self.modulo_val_id = 0
        # The entire config is copied to the save model directory hence not logged here.
        self.writer.add_text("Model", self.vis_args['model'])
        self.writer.add_text("Dataset/Train_Length", str(len(self.vis_args['train_dataset_len'])))
        self.writer.add_text("Dataset/Valid_Length", str(len(self.vis_args['valid_dataset_len'])))

    def iteration_update(self, **kwargs):
        if 'modulo' in kwargs:
            modulo = kwargs['modulo']
        else:
            modulo = None

        if kwargs['training']:
            self.writer.add_scalar("Training/QGen Batch Loss", kwargs['qgen_loss'], self.train_id)
            if modulo is not None:
                if kwargs['epoch']%modulo == 0:
                    self.writer.add_scalar("Training/Total Batch Loss", kwargs['loss'], self.modulo_train_id)
                    self.writer.add_scalar("Training/Guesser Batch Loss", kwargs['guesser_loss'], self.modulo_train_id)
                    self.writer.add_scalar("Training/Decider Batch Loss", kwargs['decider_loss'], self.modulo_train_id)
                    self.writer.add_scalar("Training/Guesser Batch Accuracy", kwargs['guesser_accuracy'], self.modulo_train_id)
                    self.writer.add_scalar("Training/Ask Batch Accuracy", kwargs['ask_accuracy'], self.modulo_train_id)
                    self.writer.add_scalar("Training/Guess Batch Accuracy", kwargs['guess_accuracy'], self.modulo_train_id)
                    self.modulo_train_id += 1
            else:
                self.writer.add_scalar("Training/Total Batch Loss", kwargs['loss'], self.train_id)
                self.writer.add_scalar("Training/Guesser Batch Loss", kwargs['guesser_loss'], self.train_id)
                self.writer.add_scalar("Training/Decider Batch Loss", kwargs['decider_loss'], self.train_id)
                self.writer.add_scalar("Training/Guesser Batch Accuracy", kwargs['guesser_accuracy'], self.train_id)
                self.writer.add_scalar("Training/Ask Batch Accuracy", kwargs['ask_accuracy'], self.train_id)
                self.writer.add_scalar("Training/Guess Batch Accuracy", kwargs['guess_accuracy'], self.train_id)
            self.train_id += 1
        else:
            self.writer.add_scalar("Validation/QGen Batch Loss", kwargs['qgen_loss'], self.valid_id)
            if modulo is not None:
                if kwargs['epoch']%modulo == 0:
                    self.writer.add_scalar("Validation/Total Batch Loss", kwargs['loss'], self.modulo_val_id)
                    self.writer.add_scalar("Validation/Guesser Batch Loss", kwargs['guesser_loss'], self.modulo_val_id)
                    self.writer.add_scalar("Validation/Decider Batch Loss", kwargs['decider_loss'], self.modulo_val_id)
                    self.writer.add_scalar("Validation/Guesser Batch Accuracy", kwargs['guesser_accuracy'], self.modulo_val_id)
                    self.writer.add_scalar("Validation/Ask Batch Accuracy", kwargs['ask_accuracy'], self.modulo_val_id)
                    self.writer.add_scalar("Validation/Guess Batch Accuracy", kwargs['guess_accuracy'], self.modulo_val_id)
                    self.modulo_val_id += 1
            else:
                self.writer.add_scalar("Validation/Total Batch Loss", kwargs['loss'], self.valid_id)
                self.writer.add_scalar("Validation/Guesser Batch Loss", kwargs['guesser_loss'], self.valid_id)
                self.writer.add_scalar("Validation/Decider Batch Loss", kwargs['decider_loss'], self.valid_id)
                self.writer.add_scalar("Validation/Guesser Batch Accuracy", kwargs['guesser_accuracy'], self.valid_id)
                self.writer.add_scalar("Validation/Ask Batch Accuracy", kwargs['ask_accuracy'], self.valid_id)
                self.writer.add_scalar("Validation/Guess Batch Accuracy", kwargs['guess_accuracy'], self.valid_id)
            self.valid_id += 1

    def epoch_update(self, **kwargs):
        if 'modulo' in kwargs:
            modulo = kwargs['modulo']
        else:
            modulo = None
        # Training
        self.writer.add_scalar("Training/QGen Epoch Loss", kwargs['train_qgen_loss'], kwargs['epoch'])
        if modulo is not None:
            modulo_epoch = kwargs['epoch']/modulo
            if kwargs['epoch']%modulo == 0:
                self.writer.add_scalar("Training/Total Epoch Loss", kwargs['train_loss'], modulo_epoch)
                self.writer.add_scalar("Training/Guesser Epoch Loss", kwargs['train_guesser_loss'], modulo_epoch)
                self.writer.add_scalar("Training/Decider Epoch Loss", kwargs['train_decider_loss'], modulo_epoch)
                self.writer.add_scalar("Training/Guesser Epoch Accuracy", kwargs['train_guesser_accuracy'], modulo_epoch)
                self.writer.add_scalar("Training/Ask Epoch Accuracy", kwargs['train_ask_accuracy'], modulo_epoch)
                self.writer.add_scalar("Training/Guess Epoch Accuracy", kwargs['train_guess_accuracy'], modulo_epoch)
        else:
            self.writer.add_scalar("Training/Total Epoch Loss", kwargs['train_loss'], kwargs['epoch'])
            self.writer.add_scalar("Training/Guesser Epoch Loss", kwargs['train_guesser_loss'], kwargs['epoch'])
            self.writer.add_scalar("Training/Decider Epoch Loss", kwargs['train_decider_loss'], kwargs['epoch'])
            self.writer.add_scalar("Training/Guesser Epoch Accuracy", kwargs['train_guesser_accuracy'], kwargs['epoch'])
            self.writer.add_scalar("Training/Ask Epoch Accuracy", kwargs['train_ask_accuracy'], kwargs['epoch'])
            self.writer.add_scalar("Training/Guess Epoch Accuracy", kwargs['train_guess_accuracy'], kwargs['epoch'])
        # Validation
        self.writer.add_scalar("Validation/QGen Epoch Loss", kwargs['valid_qgen_loss'], kwargs['epoch'])
        if modulo is not None:
            if kwargs['epoch']%modulo == 0:
                self.writer.add_scalar("Validation/Total Epoch Loss", kwargs['valid_loss'], modulo_epoch)
                self.writer.add_scalar("Validation/Guesser Epoch Loss", kwargs['valid_guesser_loss'], modulo_epoch)
                self.writer.add_scalar("Validation/Decider Epoch Loss", kwargs['valid_decider_loss'], modulo_epoch)
                self.writer.add_scalar("Validation/Guesser Epoch Accuracy", kwargs['valid_guesser_accuracy'], modulo_epoch)
                self.writer.add_scalar("Validation/Ask Epoch Accuracy", kwargs['valid_ask_accuracy'], modulo_epoch)
                self.writer.add_scalar("Validation/Guess Epoch Accuracy", kwargs['valid_guess_accuracy'], modulo_epoch)
        else:
            self.writer.add_scalar("Validation/Total Epoch Loss", kwargs['valid_loss'], kwargs['epoch'])
            self.writer.add_scalar("Validation/Guesser Epoch Loss", kwargs['valid_guesser_loss'], kwargs['epoch'])
            self.writer.add_scalar("Validation/Decider Epoch Loss", kwargs['valid_decider_loss'], kwargs['epoch'])
            self.writer.add_scalar("Validation/Guesser Epoch Accuracy", kwargs['valid_guesser_accuracy'], kwargs['epoch'])
            self.writer.add_scalar("Validation/Ask Epoch Accuracy", kwargs['valid_ask_accuracy'], kwargs['epoch'])
            self.writer.add_scalar("Validation/Guess Epoch Accuracy", kwargs['valid_guess_accuracy'], kwargs['epoch'])
