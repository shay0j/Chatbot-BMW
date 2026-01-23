// Function to send a message to the server and display the response
function sendMessage() {
    console.log('sendMessage function called'); // Debugging: Check if the function is called
    const userInput = document.getElementById('user-input').value.trim(); // Trim whitespace
    const chatWindow = document.getElementById('chat-window');

    // Check for empty input
    if (userInput === "") {
        console.warn('User input is empty. Message not sent.'); // Debugging: Warn about empty input
        return; // Exit if the input is empty
    }

    // Display user message
    chatWindow.innerHTML += `<div class="message user-message">${userInput}</div>`;

    // Send the user message to the server (Flask)
    fetch('/send', {
        method: 'POST',
        body: JSON.stringify({ message: userInput }),
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then((response) => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`); // Handle HTTP errors
        }
        return response.json();
    })
    .then((data) => {
        console.log('Received data from server:', data); // Debugging: Check server response
        // Display the bot's response
        chatWindow.innerHTML += `<div class="message bot-message">${data.response}</div>`;
        // Scroll to the bottom of the chat window to show the latest message
        chatWindow.scrollTop = chatWindow.scrollHeight;
    })
    .catch((error) => {
        console.error('Error sending message:', error);
    });

    // Clear the user input field
    document.getElementById('user-input').value = '';
}

// Attach the sendMessage function to the Send button
document.getElementById('send-button').addEventListener('click', () => {
    console.log('Send button clicked'); // Debugging: Check if the button click event is detected
    sendMessage();
});

// Allow sending a message by pressing the Enter key
document.getElementById('user-input').addEventListener('keyup', (event) => {
    if (event.key === 'Enter') {
        console.log('Enter key pressed'); // Debugging: Check if the Enter key event is detected
        sendMessage();
    }
});
