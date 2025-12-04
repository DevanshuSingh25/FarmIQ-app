// Translation keys and messages for the application
export type Language = 'en' | 'hi' | 'pa';

export interface Translations {
    // Common
    login: string;
    register: string;
    logout: string;
    email: string;
    password: string;
    submit: string;
    cancel: string;

    // Login page
    loginTitle: string;
    loginDescription: string;
    loginSuccess: string;
    welcomeBack: string;
    farmer: string;
    vendor: string;
    admin: string;
    emailRequired: string;
    passwordRequired: string;
    continueWithGoogle: string;
    continueWithPhone: string;
    dontHaveAccount: string;

    // Register page
    registerTitle: string;
    registerDescription: string;
    fullName: string;
    phone: string;
    accountCreated: string;
    accountCreatedMessage: string;
    alreadyHaveAccount: string;
    createAccount: string;

    // Role descriptions
    farmerDesc: string;
    vendorDesc: string;
    adminDesc: string;
}

export const translations: Record<Language, Translations> = {
    en: {
        // Common
        login: 'Login',
        register: 'Register',
        logout: 'Logout',
        email: 'Email',
        password: 'Password',
        submit: 'Submit',
        cancel: 'Cancel',

        // Login page
        loginTitle: 'Login',
        loginDescription: 'Sign in to your FarmIQ account',
        loginSuccess: 'Login Successful',
        welcomeBack: 'Welcome back!',
        farmer: 'Farmer',
        vendor: 'Vendor',
        admin: 'Admin',
        emailRequired: 'Email is required',
        passwordRequired: 'Password is required',
        continueWithGoogle: 'Continue with Google',
        continueWithPhone: 'Continue with Phone Number',
        dontHaveAccount: "Don't have an account?",

        // Register page
        registerTitle: 'Join the FarmIQ ecosystem',
        registerDescription: 'Choose your role and unlock tailored eco-driven insights',
        fullName: 'Full name',
        phone: 'Phone number',
        accountCreated: 'Account Created',
        accountCreatedMessage: 'Your account has been created successfully. Please login to continue.',
        alreadyHaveAccount: 'Already have an account?',
        createAccount: 'Create Account',

        // Role descriptions
        farmerDesc: 'Manage crops & field data',
        vendorDesc: 'Connect with farmers & orders',
        adminDesc: 'Oversee platform operations',
    },
    hi: {
        // Common
        login: 'लॉग इन करें',
        register: 'पंजीकरण करें',
        logout: 'लॉग आउट',
        email: 'ईमेल',
        password: 'पासवर्ड',
        submit: 'जमा करें',
        cancel: 'रद्द करें',

        // Login page
        loginTitle: 'लॉग इन करें',
        loginDescription: 'अपने FarmIQ खाते में साइन इन करें',
        loginSuccess: 'लॉगिन सफल',
        welcomeBack: 'वापसी पर स्वागत है!',
        farmer: 'किसान',
        vendor: 'विक्रेता',
        admin: 'प्रशासक',
        emailRequired: 'ईमेल आवश्यक है',
        passwordRequired: 'पासवर्ड आवश्यक है',
        continueWithGoogle: 'Google के साथ जारी रखें',
        continueWithPhone: 'फ़ोन नंबर के साथ जारी रखें',
        dontHaveAccount: 'खाता नहीं है?',

        // Register page
        registerTitle: 'FarmIQ पारिस्थितिकी तंत्र में शामिल हों',
        registerDescription: 'अपनी भूमिका चुनें और अनुकूलित पर्यावरण-संचालित अंतर्दृष्टि अनलॉक करें',
        fullName: 'पूरा नाम',
        phone: 'फ़ोन नंबर',
        accountCreated: 'खाता बनाया गया',
        accountCreatedMessage: 'आपका खाता सफलतापूर्वक बनाया गया है। कृपया जारी रखने के लिए लॉगिन करें।',
        alreadyHaveAccount: 'पहले से खाता है?',
        createAccount: 'खाता बनाएं',

        // Role descriptions
        farmerDesc: 'फसलों और खेत डेटा का प्रबंधन करें',
        vendorDesc: 'किसानों और ऑर्डर से जुड़ें',
        adminDesc: 'प्लेटफ़ॉर्म संचालन की देखरेख करें',
    },
    pa: {
        // Common
        login: 'ਲਾਗਿਨ ਕਰੋ',
        register: 'ਰਜਿਸਟਰ ਕਰੋ',
        logout: 'ਲਾਗਆਉਟ',
        email: 'ਈਮੇਲ',
        password: 'ਪਾਸਵਰਡ',
        submit: 'ਜਮ੍ਹਾਂ ਕਰੋ',
        cancel: 'ਰੱਦ ਕਰੋ',

        // Login page
        loginTitle: 'ਲਾਗਇਨ',
        loginDescription: 'ਆਪਣੇ FarmIQ ਖਾਤੇ ਵਿੱਚ ਸਾਈਨ ਇਨ ਕਰੋ',
        loginSuccess: 'ਲਾਗਇਨ ਸਫਲ',
        welcomeBack: 'ਵਾਪਸੀ ਤੇ ਸਵਾਗਤ ਹੈ!',
        farmer: 'ਕਿਸਾਨ',
        vendor: 'ਵਿਕਰੇਤਾ',
        admin: 'ਪ੍ਰਬੰਧਕ',
        emailRequired: 'ਈਮੇਲ ਲੋੜੀਂਦੀ ਹੈ',
        passwordRequired: 'ਪਾਸਵਰਡ ਲੋੜੀਂਦਾ ਹੈ',
        continueWithGoogle: 'Google ਨਾਲ ਜਾਰੀ ਰੱਖੋ',
        continueWithPhone: 'ਫ਼ੋਨ ਨੰਬਰ ਨਾਲ ਜਾਰੀ ਰੱਖੋ',
        dontHaveAccount: 'ਖਾਤਾ ਨਹੀਂ ਹੈ?',

        // Register page
        registerTitle: 'FarmIQ ਵਾਤਾਵਰਣ ਵਿੱਚ ਸ਼ਾਮਲ ਹੋਵੋ',
        registerDescription: 'ਆਪਣੀ ਭੂਮਿਕਾ ਚੁਣੋ ਅਤੇ ਅਨੁਕੂਲਿਤ ਈਕੋ-ਸੰਚਾਲਿਤ ਸੂਝ ਨੂੰ ਅਨਲੌਕ ਕਰੋ',
        fullName: 'ਪੂਰਾ ਨਾਮ',
        phone: 'ਫ਼ੋਨ ਨੰਬਰ',
        accountCreated: 'ਖਾਤਾ ਬਣਾਇਆ ਗਿਆ',
        accountCreatedMessage: 'ਤੁਹਾਡਾ ਖਾਤਾ ਸਫਲਤਾਪੂਰਵਕ ਬਣਾਇਆ ਗਿਆ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਜਾਰੀ ਰੱਖਣ ਲਈ ਲਾਗਇਨ ਕਰੋ।',
        alreadyHaveAccount: 'ਪਹਿਲਾਂ ਤੋਂ ਖਾਤਾ ਹੈ?',
        createAccount: 'ਖਾਤਾ ਬਣਾਓ',

        // Role descriptions
        farmerDesc: 'ਫਸਲਾਂ ਅਤੇ ਖੇਤ ਡੇਟਾ ਦਾ ਪ੍ਰਬੰਧਨ ਕਰੋ',
        vendorDesc: 'ਕਿਸਾਨਾਂ ਅਤੇ ਆਰਡਰਾਂ ਨਾਲ ਜੁੜੋ',
        adminDesc: 'ਪਲੇਟਫਾਰਮ ਸੰਚਾਲਨ ਦੀ ਨਿਗਰਾਨੀ ਕਰੋ',
    },
};
