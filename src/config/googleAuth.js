import {GoogleSignin} from '@react-native-google-signin/google-signin';
import {getApp} from '@react-native-firebase/app';
import {
  getAuth,
  GoogleAuthProvider,
  signInWithCredential,
} from '@react-native-firebase/auth';

const WEB_CLIENT_ID = '628864186352-0mdc5cph69v6jr800mi08to46fpsfqn6.apps.googleusercontent.com';

export const configureGoogleSignIn = () => {
  GoogleSignin.configure({
    webClientId: WEB_CLIENT_ID,
    offlineAccess: true,
  });
};

export const signInWithGoogle = async () => {
  try {
    await GoogleSignin.hasPlayServices({showPlayServicesUpdateDialog: true});

    const signInResult = await GoogleSignin.signIn();

    let idToken = signInResult.data?.idToken || signInResult.idToken;
    if (!idToken) {
      throw new Error('No ID token found');
    }

    const googleCredential = GoogleAuthProvider.credential(idToken);
    const userCredential = await signInWithCredential(getAuth(getApp()), googleCredential);

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
    try {
      const {signOut} = await import('@react-native-firebase/auth');
      await signOut(getAuth(getApp()));
    } catch (e) {}
    try {
      await GoogleSignin.signOut();
    } catch (e) {}
    return {success: true};
  } catch (error) {
    return {success: false, error: error.message};
  }
};

export const getCurrentUser = () => {
  return getAuth(getApp()).currentUser;
};

export const onAuthStateChanged = callback => {
  const {onAuthStateChanged: rnfOnAuthStateChanged} = require('@react-native-firebase/auth');
  return rnfOnAuthStateChanged(getAuth(getApp()), callback);
};
