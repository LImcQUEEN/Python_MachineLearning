def player(prev_play, opponent_history=[], transition={}):
    """
    Rock-Paper-Scissors player using a simple Markov Chain prediction.

    - prev_play: last move of the opponent
    - opponent_history: list of previous opponent moves (state memory)
    - transition: dictionary to store transition frequencies between moves
    """
    moves = ['R', 'P', 'S']
    counter = {'R': 'P', 'P': 'S', 'S': 'R'}  # what beats what

    # Initialize first move randomly if no history
    if not prev_play:
        next_move = 'R'  # could also pick randomly here
    else:
        # Update opponent history
        opponent_history.append(prev_play)

        # Update transition table
        if len(opponent_history) >= 2:
            prev = opponent_history[-2]
            if prev not in transition:
                transition[prev] = {'R': 0, 'P': 0, 'S': 0}
            transition[prev][prev_play] += 1

        # Predict next move based on last move
        last_move = opponent_history[-1]
        if last_move in transition:
            # Pick opponent's most likely next move
            predicted_move = max(transition[last_move], key=transition[last_move].get)
            # Play what beats the predicted move
            next_move = counter[predicted_move]
        else:
            # If no data, default to countering most frequent
            freq = {'R': 0, 'P': 0, 'S': 0}
            for move in opponent_history:
                freq[move] += 1
            predicted = max(freq, key=freq.get)
            next_move = counter[predicted]

    return next_move
