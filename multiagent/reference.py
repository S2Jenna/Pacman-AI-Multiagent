return self.MinimaxSearch(gameState, 1, 0 )
        #action, score = self.minimax(0, 0, gameState)  # Get the action and score for pacman (agent_index=0)
        #return action  # Return the action to be done as per minimax algorithm
        #util.raiseNotDefined()

    def MinimaxSearch(self, gameState, currentDepth, agentIndex):
        "terminal check"
        if currentDepth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
    
        "minimax algorithm"
        legalMoves = [action for action in gameState.getLegalActions(agentIndex) if action!='Stop']
    
        # update next depth
        nextIndex = agentIndex + 1
        nextDepth = currentDepth
        if nextIndex >= gameState.getNumAgents():
            nextIndex = 0
            nextDepth += 1
    
        # Choose one of the best actions or keep query the minimax result
        results = [self.MinimaxSearch( gameState.generateSuccessor(agentIndex, action) ,\
                                      nextDepth, nextIndex) for action in legalMoves]
        if agentIndex == 0 and currentDepth == 1: # pacman first move
            bestMove = max(results)
            bestIndices = [index for index in range(len(results)) if results[index] == bestMove]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            #print 'pacman %d' % bestMove
            return legalMoves[chosenIndex]
    
        if agentIndex == 0:
            bestMove = max(results)
            #print bestMove
            return bestMove
        else:
            bestMove = min(results)
            #print bestMove
            return bestMove