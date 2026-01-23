// firebase.js
// Import the Firebase SDKs
import { initializeApp } from 'firebase/app';
import { getAnalytics } from 'firebase/analytics'; // Import Firebase Analytics

// Your web app's Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyCcHpRSv8aXyoV4PlIyo0ml6bBKBhEjhrc",
    authDomain: "eric-c1b87.firebaseapp.com",
    databaseURL: "https://eric-c1b87-default-rtdb.europe-west1.firebasedatabase.app",
    projectId: "eric-c1b87",
    storageBucket: "eric-c1b87.appspot.com",
    messagingSenderId: "637336074053",
    appId: "1:637336074053:web:386ae8c463d014e920b02c",
    measurementId: "G-THWKZJZHQ9"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

export { app, analytics };
