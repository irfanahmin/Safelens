document.addEventListener('DOMContentLoaded', () => {
    const liveFeed = document.getElementById('liveFeed');
    const launchButton = document.querySelector('.launch-button');
    let isCameraOn = false; // Variable to track if the camera is on

    // Function to start the camera
    const startCamera = () => {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                liveFeed.srcObject = stream;
                liveFeed.play();
                isCameraOn = true;
                launchButton.textContent = 'Stop'; // Change button text to 'Stop'
            })
            .catch(error => {
                console.error('Error accessing camera:', error);
            });
    };

    // Function to stop the camera
    const stopCamera = () => {
        const mediaStream = liveFeed.srcObject;
        const tracks = mediaStream.getTracks();
        tracks.forEach(track => {
            track.stop();
        });
        liveFeed.srcObject = null;
        isCameraOn = false;
        launchButton.textContent = 'Launch'; // Change button text back to 'Launch'
    };

    // Click event handler for the launch button
    launchButton.addEventListener('click', () => {
        if (!isCameraOn) {
            startCamera();
        } else {
            stopCamera();
        }
    });

    // Hover effect for the launch button
    launchButton.addEventListener('mouseenter', () => {
        launchButton.style.backgroundColor = '#cf7b5a'; // Darker shade on hover
    });

    // Restore button color on mouse leave
    launchButton.addEventListener('mouseleave', () => {
        launchButton.style.backgroundColor = '#e08d6e'; // Original color on mouse leave
    });
});