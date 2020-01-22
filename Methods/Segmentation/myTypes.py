import enum

class LearningStep(enum.Enum):
    TRAINING = 0
    TEST = 1
    VALIDATION = 2

class TrainTestSplit():
    train_percentage = None
    test_percentage = None
    validation_percentage = None
    def __init__(self, train_percentage=0.7, test_percentage=0.15):
        self.train_percentage = train_percentage
        self.test_percentage = test_percentage
        self.validation_percentage = 1 - self.test_percentage - self.train_percentage

    def getYSplit(self,Height):
        return (self.train_percentage*Height, (self.train_percentage+self.test_percentage)*Height)
