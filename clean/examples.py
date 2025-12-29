ids = {
        'en' : [1,25,117],
        'pe' : [36,80,161],
        'cross' : [6,104,239],
        # 'ar' : [5,29,121]
        }

en = {
    ids["en"][0] : { 
        "primary_clues": [
            "The central image is the flag of France.",
            "The flag is enclosed or surrounded by a circle.",
            "The prefix for 'around' or 'surrounding' is 'Circum-'.",
            "Combining the prefix with the country name creates 'Circum-France'.",
            "Phonetically, this sounds like the geometric term 'Circumference'."
        ],
        "candidates": [
            "France Circles", 
            "Rounded France", 
            "Circumference"  
        ],
    },
    ids["en"][1] : {
        "primary_clues": [
            "A man sits in a corner with tattered clothes and a bowl containing broken scraps, symbolizing having lost everything.",
            "His desolate condition and the empty/broken contents of his begging bowl indicate he is 'in need'.",
            "A thought bubble above his head displays the letter 'L'.",
            "Combining his condition ('Need') with the letter ('L') creates the construction 'Need-L'.",
            "Phonetically, this sounds like the word 'Needle'."
        ],
        "candidates": [
            "Needs L", 
            "L Wants", 
            "Needle" 
        ],
    },
    ids["en"][2] : {
        "primary_clues": [
            "The image depicts a personified church building.",
            "The church appears sick, with a thermometer in its mouth and a tissue in its hand, indicating it is 'ill'.",
            "This combines to form 'Church' + 'ill', sounding like 'Churchill'.",
            "The church is also holding a pack of cigarettes clearly labeled 'Winston'.",
            "Combining the brand name with the other elements creates 'Winston Churchill'."
        ],
        "candidates": [
            "Sick Church Smokes",
            "Winstons Sick Home",
            "Winston Churchill"
        ],
    },
}

pe = {
    ids["pe"][0] : {
        "primary_clues": [
            "The image displays a human brain, which translates to 'Mokh' (مخ) in Persian.",
            "The visual style of the brain is blurred or faded, which translates to the adjective 'Tar' (تار).",
            "Combining the noun 'Mokh' with the adjective 'Tar' creates the construction 'Mokh-Tar'.",
            "This forms the common Persian name 'Mokhtar' (مختار)."
        ],
        "candidates": [
            "مخ مات", 
            "تارمخ",  
            "مختار" 
        ],
    },
    ids["pe"][1] : {
        "primary_clues": [
            "The image shows a courtroom setting with lawyers, representing the legal concept of 'Vekalat' (وکالت) or power of attorney.",
            "The cartoon character is Tom from 'Tom and Jerry', whose name in Persian is 'Tam' (تام).",
            "Combining the legal term 'Vekalat' with the character's name 'Tam' creates the phrase 'Vekalat-e Tam' (وکالت تام).",
            "This phrase is a common legal term meaning 'full power of attorney'."
        ],
        "candidates": [
            "تام وکیله",
            "وکیل تامی",
            "وکالت تام"     
        ],
    },
    ids["pe"][2] : {
        "primary_clues": [
            "The image displays a grilling skewer (seekh) typically used for making kebabs.",
            "Instead of meat, the skewer holds multiple instances of the Persian letter 'Che' (چ).",
            "In Persian, 'multiple Che's' or 'several Che's' translates to 'Chand Che' (چند چ).",
            "The phrase 'Chand Che' sounds phonetically very similar to 'Chenjeh' (چنجه).",
            "This creates a pun on 'Kebab Chenjeh' (Lamb Chop Kebab), replacing the meat with letters."
        ],
        "candidates": [
            "چهار کباب", 
            "کباب چندچ", 
            "کباب چنجه"  
        ],
    },
}

cross = {
    ids["cross"][0] : {
        "primary_clues": [
            "The image shows the Persian letter 'Seen' (س).",
            "Placed on top ('Roosh' - روش) of the 'Seen' are three ('Se' - سه) hats ('Hat' - هت).",
            "Combining 'Seen' + 'Roosh' creates the name 'Soroush' (سروش).",
            "Combining 'Se' (Three) + 'Hat' (English word for hat) sounds like 'Sehhat' (صحت).",
            "Together, they form the name of the famous Iranian director 'Soroush Sehhat' (سروش صحت)."
        ],
        "candidates": [
            "سه کلاه س",
            "کلاه سه س",
            "سروش صحت" 
        ],
    },
    ids["cross"][1] : {
        "primary_clues": [
            "The image shows the letter 'k' surrounded by three checkmarks (ticks).",
            "The letter 'k' is in the middle, which translates to 'Mian' (میان) in Persian.",
            "Combining 'K' + 'Mian' phonetically sounds like 'Kamion' (کامیون), meaning Truck.",
            "The 'k' is placed among ('La' - لا) the three ('Se' - سه) ticks ('Tick' - تیک).",
            "The phrase 'La-Se-Tick' (لا سه تیک) sounds like 'Lastik' (لاستیک), meaning Tire.",
            "Together, the visual pun creates 'Lastik Kamion' (لاستیک کامیون)."
        ],
        "candidates": [
            "لاستیک اتوبوس",
            "لاستیک دوچرخه",
            "لاستیک کامیون"
        ],
    },
    ids["cross"][2] : {
        "primary_clues": [
            "The image features a garden, which translates to 'Bagh' (باغ) in Persian.",
            "The letters 'DR' are superimposed on the scene.",
            "Phonetically, the English letters 'DR' (Dee-Ar) sound identical to the Persian word 'Diar' (دیار), meaning 'Land' or 'Realm'.",
            "Combining the two components yields 'Diar' + 'Bagh'.",
            "This creates a pun on the common phrase 'Diar-e Baghi' (دیار باقی), which means 'The Eternal Realm' or 'The Afterlife'.",
            "The pun works by substituting the word 'Bagh' (Garden) for the phonetically similar 'Baghi' (Eternal)."
        ],
        "candidates": [
            "دکتر باغی",
            "باغ دیوار",
            "دیار باقی"
        ],
    },
}

ar = {
    # ids["ar"][0] : "",
    # ids["ar"][1] : "",
    # ids["ar"][2] : "",
}

def load_derivations():
    return {
        'en' : en,
        'pe' : pe,
        'cross' : cross,
        # 'ar' : ar,
    }
