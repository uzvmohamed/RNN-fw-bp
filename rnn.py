import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, seed=42):

        np.random.seed(seed)
        
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output
        
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias
        self.by = np.zeros((output_size, 1))  # Output bias
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.reset_gradients()
        
    def reset_gradients(self):
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dWhy = np.zeros_like(self.Why)
        self.dbh = np.zeros_like(self.bh)
        self.dby = np.zeros_like(self.by)
        
    def forward(self, inputs):

        h_prev = np.zeros((self.hidden_size, 1))  
        self.inputs = inputs  
        self.hidden_states = [h_prev]  
        self.outputs = []  
        
        for x in inputs:
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
            
            y = np.dot(self.Why, h) + self.by
            
            self.hidden_states.append(h)
            self.outputs.append(y)
            
            h_prev = h
            
        return self.outputs, self.hidden_states
    
    def backward(self, doutputs):
    
        self.reset_gradients()

        dh_next = np.zeros((self.hidden_size, 1)) 

        for t in reversed(range(len(doutputs))):
            x = self.inputs[t]
            h = self.hidden_states[t+1]
            h_prev = self.hidden_states[t]
            dy = doutputs[t]
            
            self.dWhy += np.dot(dy, h.T)
            self.dby += dy
            
            dh = np.dot(self.Why.T, dy) + dh_next
            
            dtanh = (1 - h * h) * dh  
            
            self.dbh += dtanh
            self.dWxh += np.dot(dtanh, x.T)
            self.dWhh += np.dot(dtanh, h_prev.T)
            
            dh_next = np.dot(self.Whh.T, dtanh)
            
    def clip_gradients(self, max_norm=1.0):
        
        for grad in [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby]:
            np.clip(grad, -max_norm, max_norm, out=grad)
            
    def update_weights(self, learning_rate):
       
        self.Wxh -= learning_rate * self.dWxh
        self.Whh -= learning_rate * self.dWhh
        self.Why -= learning_rate * self.dWhy
        self.bh -= learning_rate * self.dbh
        self.by -= learning_rate * self.dby
        
    def compute_loss(self, outputs, targets):
        
        loss = 0
        for y, t in zip(outputs, targets):
            loss += np.sum((y - t) ** 2)
        return loss / len(outputs)
    
    def compute_gradients(self, outputs, targets):
       
        return [2 * (output - target) / len(outputs) 
                for output, target in zip(outputs, targets)]


def train_rnn(rnn, inputs, targets, epochs=100, learning_rate=0.01, verbose=True):
    
    for epoch in range(epochs):
        total_loss = 0
        
        for input_seq, target_seq in zip(inputs, targets):

            outputs, _ = rnn.forward(input_seq)
            
            loss = rnn.compute_loss(outputs, target_seq)
            total_loss += loss
            
            doutputs = rnn.compute_gradients(outputs, target_seq)
            
            rnn.backward(doutputs)
            
            rnn.clip_gradients(max_norm=1.0)
            
            rnn.update_weights(learning_rate)
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            avg_loss = total_loss / len(inputs)
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")


if __name__ == "__main__":
    input_size = 3
    hidden_size = 5
    output_size = 2
    rnn = SimpleRNN(input_size, hidden_size, output_size)
    
    sequences = [
        [np.array([[0.1], [0.2], [0.3]]),
         np.array([[0.4], [0.5], [0.6]]),
         np.array([[0.7], [0.8], [0.9]]),
         np.array([[1.0], [1.1], [1.2]])],
        
        [np.array([[0.3], [0.2], [0.1]]),
         np.array([[0.6], [0.5], [0.4]]),
         np.array([[0.9], [0.8], [0.7]]),
         np.array([[1.2], [1.1], [1.0]])],
        
        [np.array([[0.5], [0.0], [-0.5]]),
         np.array([[0.3], [-0.2], [-0.7]]),
         np.array([[0.1], [-0.4], [-0.9]]),
         np.array([[-0.1], [-0.6], [-1.1]])]
    ]
    
    targets = [
        [np.array([[x[0][0] + x[1][0]], [x[2][0] - x[0][0]]]) for x in seq]
        for seq in sequences
    ]
    
    print("Starting training...")
    train_rnn(rnn, sequences, targets, epochs=100, learning_rate=0.01)
    
    print("\nTesting trained RNN:")
    test_input = sequences[0]  
    outputs, _ = rnn.forward(test_input)
    
    print("\nInput Sequence:")
    for i, x in enumerate(test_input):
        print(f"Step {i}: {x.flatten()}")
    
    print("\nPredicted Outputs:")
    for i, y in enumerate(outputs):
        print(f"Step {i}: {y.flatten()}")
    
    print("\nActual Targets:")
    for i, t in enumerate(targets[0]):
        print(f"Step {i}: {t.flatten()}")