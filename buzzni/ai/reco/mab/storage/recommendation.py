class Recommendation(object):
    """The object to store a recommendation.

    Parameters
    ----------
    action : Action
    estimated_reward: float
    uncertainty: float
    score: float
    """

    def __init__(self, action, estimated_reward, uncertainty, score,
                 reward=None):
        self.action = action
        self.estimated_reward = estimated_reward
        self.uncertainty = uncertainty
        self.score = score
        self.reward = reward
