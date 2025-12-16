import {GoogleSignin} from '@react-native-google-signin/google-signin';
import auth from '@react-native-firebase/auth';
import {GOOGLE_WEB_CLIENT_ID} from '@env';

// Google Web Client ID from environment variables
const WEB_CLIENT_ID = GOOGLE_WEB_CLIENT_ID;

export const configureGoogleSignIn = () => {
  GoogleSignin.configure({
    webClientId: WEB_CLIENT_ID,
    offlineAccess: true,
  });
};

export const signInWithGoogle = async () => {
  try {
    // Check Play Services
    await GoogleSignin.hasPlayServices({showPlayServicesUpdateDialog: true});

    // Sign in with Google
    const signInResult = await GoogleSignin.signIn();

    // Get the ID token
    let idToken = signInResult.data?.idToken;
    if (!idToken) {
      throw new Error('No ID token found');
    }

    // Create Firebase credential with Google ID token
    const googleCredential = auth.GoogleAuthProvider.credential(idToken);

    // Sign in to Firebase with the credential
    const userCredential = await auth().signInWithCredential(googleCredential);

    return {
      success: true,
      user: userCredential.user,
      data: {
        email: userCredential.user.email,
        displayName: userCredential.user.displayName,
        photoURL: userCredential.user.photoURL,
        uid: userCredential.user.uid,
      },
    };
  } catch (error) {
    console.error('Google Sign-In Error:', error);
    return {success: false, error: error.message};
  }
};

export const signOutFromGoogle = async () => {
  try {
    // Sign out from Firebase (ignore if no user signed in)
    try {
      await auth().signOut();
    } catch (e) {
      // Ignore "no-current-user" error - it's expected if user used email login
    }
    // Sign out from Google (ignore if not signed in with Google)
    try {
      await GoogleSignin.signOut();
    } catch (e) {
      // Ignore - user may not have signed in with Google
    }
    return {success: true};
  } catch (error) {
    return {success: false, error: error.message};
  }
};

export const getCurrentUser = () => {
  return auth().currentUser;
};

// Listen for auth state changes
export const onAuthStateChanged = callback => {
  return auth().onAuthStateChanged(callback);
};
