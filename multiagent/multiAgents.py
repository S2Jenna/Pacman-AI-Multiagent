# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"  
        # use manhattan distance to calculate the evaluation score
        # foodlist gets and stores all the food in current game sate 
        foodList = list(successorGameState.getPacmanPosition())
        # define max negaive score when the plater loses
        min = 999999999
        dist = 0
        curfood = currentGameState.getFood()
        # curfoodlist gets the curfood as a list
        curfoodlist = curfood.asList()
        # calculate the minimum distance to each food by iterating the foodlist
        for i in range(len(curfoodlist)) :
            dist = (manhattanDistance(curfoodlist[i], foodList))
            if dist < min: 
                min = dist # take the minimum distance as the return value
        min = -min # make mindist to negative 
        
        for state in newGhostStates: # iterating the newGhoststates
            if state.scaredTimer == 0 and state.getPosition() == tuple(foodList): # if the player meets ghost
                return -999999999 # returns the max negative score
            
        if action == 'Stop': # if 'Stop' appears in the action list
            return -999999999 # returns the max negative score
        return min # o.w return mindist"""

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # for pacman (agentIndex=0)
        action, score = self.minimax(gameState, 0, 0) # done the minmax algorithm
        return action  # Return the action from per algorithm

    def minimax(self, gameState, depth, agentIndex):
        
        # run once all agents have finished playing their turn in a move
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0 ## indicates Pacman's move
            depth += 1

        # returns the evaluationFunction when the maximum depth has been reached
        if depth == self.depth:
            return None, self.evaluationFunction(gameState)
        
        ## initializes vars for the ideal action & score for the agents
        ## Pacman ideally aims for the maximum score, while 
        ## the ghosts ideally aim for the minimum score
        BestAction = None
        BestScore = None

        # max player (agentIndex = 0): the best score is max score among its successor states
        # min player (agentIndex !=0): the best score is min score among its successor states
        if agentIndex == 0:  #Pacman's turn

            ## recursively calls minimax in order to get Pacman's ideal score
            for action in gameState.getLegalActions(agentIndex):

                ## retrieves the minimax score out of all possible actions & increments agentIndex
                ## to allow the ghosts to move
                nextState = gameState.generateSuccessor(agentIndex, action)
                _, score = self.minimax(nextState, depth, agentIndex + 1)

                # Update the best score and action
                if BestScore == None or score > BestScore:
                    BestScore = score
                    BestAction = action

        else:  # the ghost turn
            for action in gameState.getLegalActions(agentIndex): 
                nextState = gameState.generateSuccessor(agentIndex, action) # gets the minimax score of successor
                _, score = self.minimax(nextState, depth, agentIndex + 1) # Increments agentIndex by 1

                # Update the best score and action, if best score is None (not updated yet) or if current score is
                # better than the best score found so far
                if BestScore == None or score < BestScore:
                    BestScore = score
                    BestAction = action

        # returns evaluationFunction if using a leaf node (no successor states present)
        if BestScore == None:
            return None, self.evaluationFunction(gameState) # return evaluationFunction
        return BestAction, BestScore  # Return the best_action and best_score
      

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ## represents alpha & beta - initialized to infinity
        inf = float('inf')
        action, score = self.alphaBeta(gameState, 0, 0, -inf, inf)
        return action
        
    def alphaBeta(self, gameState, depth, agentIndex, alpha, beta):
        # run once all agents have finished playing their turn in a move
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0 ## indicates Pacman's move
            depth += 1

        # returns the evaluationFunction when the maximum depth has been reached
        if depth == self.depth:
            return None, self.evaluationFunction(gameState)
        
        ## initializes vars for the ideal action & score for the agents
        ## Pacman ideally aims for the maximum score, while 
        ## the ghosts ideally aim for the minimum score
        BestAction = None
        BestScore = None

        ## for Pacman's turn
        if agentIndex == 0:
            for action in gameState.getLegalActions(agentIndex):
                # recursively calls alphaBeta in order to get Pacman's ideal score
                ## retrieves the minimax score out of all possible actions & increments agentIndex
                ## to allow the ghosts to move
                nextState = gameState.generateSuccessor(agentIndex, action)
                _, score = self.alphaBeta(nextState, depth, agentIndex+1, alpha, beta)

                # Update the best score and action
                if BestScore == None or score > BestScore:
                    BestScore = score
                    BestAction = action
                
                ## updates the alpha score
                alpha = max(alpha, score)

                ## prunes tree if alpha > beta
                if alpha > beta: break

        else: ## for ghost's turn
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                _, score = self.alphaBeta(nextState, depth, agentIndex+1, alpha, beta)

                ## updates the ideal Action & Score for the ghost
                if BestScore == None or score < BestScore:
                    BestAction = action
                    BestScore = score
                
                ## updates the beta score
                beta = min(beta, score)

                ## prunes tree if beta < alpha
                if beta < alpha: break

        ## returns evaluatorFuction for those w no successor states
        if BestScore == None:
            return None, self.evaluationFunction(gameState)
        
        return BestAction, BestScore
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, score = self.ExpectiMax(gameState, 0, 0)
        return action
    
    def ExpectiMax(self, gameState, depth, agentIndex):

        # run once all agents have finished playing their turn in a move
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0 ## indicates Pacman's move
            depth += 1
        
        ## returns once max depth has been reached
        if depth == self.depth:
            return None, self.evaluationFunction(gameState)
        
        ## initialize ideal action & score vars for agents
        BestAction = None
        BestScore = None

        ## for Pacman's turn
        if agentIndex == 0:
            for action in gameState.getLegalActions(agentIndex): ## goes through all of Pacman's available moves
                nextState = gameState.generateSuccessor(agentIndex, action)
                _, score = self.ExpectiMax(nextState, depth, agentIndex+1)

                if BestScore == None or score > BestScore:
                    BestAction = action
                    BestScore = score

        else: ## for ghost's turn
            ghostActions = len(gameState.getLegalActions(agentIndex))
            if ghostActions != 0: ## if the ghost has possible moves
                ## probability for updating the total score with all possible ghostActions
                probability = 1.0 / ghostActions
            
            for action in gameState.getLegalActions(agentIndex): ## all of the ghost's possible actions
                nextState = gameState.generateSuccessor(agentIndex, action)
                _, score = self.ExpectiMax(nextState, depth, agentIndex+1)

                if BestScore == None: BestScore = 0.0
                BestScore += probability * score
                BestAction = action
        
        if BestScore == None:
            return None, self.evaluationFunction(gameState)
        return BestAction, BestScore

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
     # Useful information you can extract from a GameState (pacman.py)
    ""
    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood()
    curGhostStates = currentGameState.getGhostStates()

    FOOD_COEF, GHOST_COEF, SCARED_COEF = 10.0, 10.0, 200.0

    # curScore updates the game score
    curScore = currentGameState.getScore() # each move deducts one point from score

    ghostScore = 0
    # calculates the distance from Pacman to ghosts by iterating curGhostStates
    for ghost in curGhostStates:
        closestGhost = manhattanDistance(curPos, curGhostStates[0].getPosition()) # use Manhattan distance to get the closest ghost
        if closestGhost > 0: # if there are ghosts in the map
            if ghost.scaredTimer > 0:  # if ghost is scared
                ghostScore += SCARED_COEF / closestGhost # go to ghost
            else:  # ghost isn't scared
                ghostScore -= GHOST_COEF / closestGhost # avoid ghost
    curScore += ghostScore

    foodList = curFood.asList()
    closestFood = [manhattanDistance(curPos, x) for x in foodList] # use Manhattan distance to calculate the closest food
    if len(closestFood) != 0:
        curScore += FOOD_COEF / min(closestFood)

    return curScore

class valAction:

  def __init__(self, val, action):
    self.val = val
    self.action = action
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
