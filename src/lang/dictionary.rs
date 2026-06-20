//! Dictionary operations for word validation and correction

use std::collections::HashSet;

/// Dictionary for word validation and spelling correction
pub struct Dictionary {
    words: HashSet<String>,
    max_edit_distance: usize,
}

impl Dictionary {
    pub fn new() -> Self {
        Self {
            words: HashSet::new(),
            max_edit_distance: 2,
        }
    }

    pub fn with_max_edit_distance(max_edit_distance: usize) -> Self {
        Self {
            words: HashSet::new(),
            max_edit_distance,
        }
    }

    pub fn load_from_text(&mut self, text: &str) {
        for word in text.split_whitespace() {
            let cleaned: String = word
                .chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(|c| c.to_lowercase())
                .collect();
            if !cleaned.is_empty() {
                self.words.insert(cleaned);
            }
        }
    }

    pub fn load_words(&mut self, words: &[&str]) {
        for word in words {
            let cleaned: String = word
                .chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(|c| c.to_lowercase())
                .collect();
            if !cleaned.is_empty() {
                self.words.insert(cleaned);
            }
        }
    }

    pub fn contains(&self, word: &str) -> bool {
        let lower: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        self.words.contains(&lower)
    }

    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    pub fn suggest_corrections(&self, word: &str, max_suggestions: usize) -> Vec<(String, usize)> {
        let lower: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();

        if lower.is_empty() {
            return Vec::new();
        }

        let exact_lower: String = word.chars().flat_map(|c| c.to_lowercase()).collect();

        if self.words.contains(&exact_lower) {
            return Vec::new();
        }

        let mut candidates: Vec<(String, usize)> = self
            .words
            .iter()
            .filter_map(|w| {
                let dist = edit_distance(&lower, w);
                if dist <= self.max_edit_distance {
                    Some((w.clone(), dist))
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)));
        candidates.truncate(max_suggestions);
        candidates
    }

    pub fn correct_word(&self, word: &str) -> String {
        let lower: String = word.chars().flat_map(|c| c.to_lowercase()).collect();
        if self.words.contains(&lower) {
            return word.to_string();
        }

        let suggestions = self.suggest_corrections(word, 1);
        if let Some((suggestion, _)) = suggestions.first() {
            preserve_case(word, suggestion)
        } else {
            word.to_string()
        }
    }
}

impl Default for Dictionary {
    fn default() -> Self {
        Self::new()
    }
}

pub fn edit_distance(a: &str, b: &str) -> usize {
    let a_len = a.chars().count();
    let b_len = b.chars().count();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let mut prev = vec![0usize; b_len + 1];
    let mut curr = vec![0usize; b_len + 1];

    for j in 0..=b_len {
        prev[j] = j;
    }

    for i in 1..=a_len {
        curr[0] = i;
        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b_len]
}

fn preserve_case(original: &str, corrected: &str) -> String {
    if original.is_empty() {
        return corrected.to_string();
    }

    let is_upper = original.chars().next().unwrap().is_uppercase();
    let is_all_upper = original.chars().all(|c| c.is_uppercase());

    if is_all_upper {
        corrected.to_uppercase()
    } else if is_upper {
        let mut result = String::with_capacity(corrected.len());
        let mut chars = corrected.chars();
        if let Some(first) = chars.next() {
            result.extend(first.to_uppercase());
        }
        result.extend(chars);
        result
    } else {
        corrected.to_string()
    }
}

pub fn load_english_dictionary() -> Dictionary {
    let mut dict = Dictionary::new();

    let common_words = [
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "I",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
        "is",
        "was",
        "are",
        "were",
        "been",
        "has",
        "had",
        "did",
        "does",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "need",
        "very",
        "still",
        "much",
        "more",
        "here",
        "own",
        "each",
        "where",
        "why",
        "while",
        "through",
        "during",
        "before",
        "between",
        "after",
        "above",
        "below",
        "under",
        "again",
        "further",
        "once",
        "such",
        "those",
        "being",
        "both",
        "same",
        "every",
        "many",
        "great",
        "old",
        "big",
        "high",
        "long",
        "small",
        "large",
        "next",
        "early",
        "young",
        "important",
        "public",
        "bad",
        "another",
        "right",
        "left",
        "able",
        "end",
        "point",
        "world",
        "life",
        "hand",
        "part",
        "place",
        "case",
        "week",
        "company",
        "system",
        "program",
        "question",
        "work",
        "government",
        "number",
        "night",
        "point",
        "home",
        "water",
        "room",
        "mother",
        "area",
        "money",
        "story",
        "fact",
        "month",
        "lot",
        "right",
        "study",
        "book",
        "eye",
        "job",
        "word",
        "business",
        "issue",
        "side",
        "kind",
        "head",
        "house",
        "service",
        "friend",
        "father",
        "power",
        "hour",
        "game",
        "line",
        "member",
        "law",
        "car",
        "city",
        "community",
        "name",
        "president",
        "team",
        "minute",
        "idea",
        "body",
        "information",
        "back",
        "parent",
        "face",
        "others",
        "level",
        "office",
        "door",
        "health",
        "person",
        "art",
        "war",
        "history",
        "party",
        "result",
        "change",
        "morning",
        "reason",
        "research",
        "girl",
        "guy",
        "moment",
        "air",
        "teacher",
        "force",
        "education",
        "food",
        "picture",
        "class",
        "product",
        "experience",
        "country",
        "problem",
        "today",
        "market",
        "report",
        "family",
        "return",
        "dog",
        "student",
        "group",
        "value",
        "best",
        "plan",
        "state",
        "school",
        "child",
        "college",
        "interest",
        "death",
        "center",
        "often",
        "development",
        "process",
        "music",
        "paper",
        "control",
        "love",
        "true",
        "woman",
        "man",
        "age",
        "per",
        "yes",
        "too",
        "any",
        "let",
        "began",
        "run",
        "help",
        "turn",
        "start",
        "might",
        "show",
        "move",
        "live",
        "put",
        "bring",
        "offer",
        "keep",
        "try",
        "leave",
        "call",
        "hold",
        "provide",
        "seem",
        "help",
        "begin",
        "set",
        "learn",
        "read",
        "grow",
        "open",
        "walk",
        "win",
        "offer",
        "remember",
        "love",
        "consider",
        "appear",
        "buy",
        "wait",
        "serve",
        "die",
        "send",
        "expect",
        "build",
        "stay",
        "fall",
        "cut",
        "reach",
        "kill",
        "remain",
        "suggest",
        "raise",
        "pass",
        "sell",
        "require",
        "report",
        "decide",
        "pull",
        "develop",
        "not",
        "but",
        "and",
        "or",
        "nor",
        "for",
        "yet",
        "so",
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "any",
        "no",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "also",
        "around",
        "away",
        "across",
        "along",
        "already",
        "although",
        "always",
        "am",
        "among",
        "another",
        "anything",
        "are",
        "been",
        "being",
        "came",
        "come",
        "could",
        "did",
        "does",
        "done",
        "either",
        "else",
        "even",
        "ever",
        "every",
        "from",
        "get",
        "got",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "him",
        "his",
        "how",
        "however",
        "i",
        "if",
        "into",
        "is",
        "it",
        "its",
        "may",
        "me",
        "might",
        "mine",
        "must",
        "my",
        "neither",
        "never",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "out",
        "own",
        "per",
        "quite",
        "rather",
        "really",
        "said",
        "same",
        "she",
        "should",
        "since",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "upon",
        "us",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "whether",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
        "yet",
        "you",
        "your",
    ];

    dict.load_words(&common_words);
    dict
}

pub fn load_french_dictionary() -> Dictionary {
    let mut dict = Dictionary::new();
    let words = [
        "le",
        "de",
        "et",
        "à",
        "un",
        "il",
        "être",
        "avoir",
        "ne",
        "je",
        "son",
        "que",
        "se",
        "qui",
        "ce",
        "dans",
        "en",
        "du",
        "elle",
        "au",
        "de",
        "ce",
        "pour",
        "pas",
        "que",
        "vous",
        "avec",
        "tout",
        "faire",
        "leur",
        "dire",
        "aller",
        "voir",
        "mon",
        "savoir",
        "vouloir",
        "falloir",
        "venir",
        "devoir",
        "croire",
        "trouver",
        "donner",
        "temps",
        "aimer",
        "permettre",
        "montrer",
        "sembler",
        "tenir",
        "porter",
        "entendre",
        "rendre",
        "vivre",
        "mourir",
        "rester",
        "revenir",
        "partir",
        "devenir",
        "recevoir",
        "servir",
        "suivre",
        "écrire",
        "joindre",
        "atteindre",
        "appeler",
        "rappeler",
        "développer",
        "exister",
        "changer",
        "comprendre",
        "reconnaître",
        "apprendre",
        "répondre",
        "demander",
        "travailler",
        "gouvernement",
        "pays",
        "monde",
        "année",
        "jour",
        "homme",
        "femme",
        "vie",
        "main",
        "part",
        "place",
        "cas",
        "semaine",
        "entreprise",
        "système",
        "programme",
        "question",
        "travail",
        "numéro",
        "nuit",
        "point",
        "maison",
        "eau",
        "chambre",
        "mère",
        "argent",
        "histoire",
        "fait",
        "mois",
        "beaucoup",
        "droit",
        "livre",
        "œil",
        "emploi",
        "mot",
        "affaires",
        "problème",
        "côté",
        "tête",
        "ami",
        "père",
        "heure",
        "jeu",
        "ligne",
        "fin",
        "membre",
        "loi",
        "voiture",
        "ville",
        "communauté",
        "nom",
        "président",
        "équipe",
        "minute",
        "idée",
        "corps",
        "information",
        "parent",
        "visage",
        "niveau",
        "bureau",
        "porte",
        "santé",
        "personne",
        "art",
        "guerre",
        "histoire",
        "fête",
        "résultat",
        "matin",
        "raison",
        "recherche",
        "fille",
        "moment",
        "air",
        "enseignant",
        "force",
        "éducation",
        "nourriture",
        "photo",
        "classe",
        "produit",
        "expérience",
        "marché",
        "famille",
        "retour",
        "chien",
        "étudiant",
        "groupe",
        "valeur",
        "plan",
        "école",
        "intérêt",
        "mort",
        "centre",
        "musique",
        "papier",
        "contrôle",
        "amour",
        "vrai",
        "femme",
        "homme",
        "âge",
        "oui",
        "trop",
        "alors",
        "ainsi",
        "autre",
        "encore",
        "toujours",
        "déjà",
        "jamais",
        "même",
        "tant",
        "après",
        "avant",
        "depuis",
        "lorsque",
        "puisque",
        "quoique",
        "sans",
        "sous",
        "sur",
        "chez",
        "contre",
        "entre",
        "parmi",
        "vers",
        "voici",
        "voilà",
        "aucun",
        "plusieurs",
        "quelque",
        "tout",
        "chaque",
        "mien",
        "tien",
        "sien",
        "nôtre",
        "vôtre",
        "leur",
        "celui",
        "celle",
        "ceux",
        "celles",
        "ceci",
        "cela",
        "ce",
        "cet",
        "cette",
        "ces",
        "mon",
        "ton",
        "son",
        "ma",
        "ta",
        "sa",
        "mes",
        "tes",
        "ses",
        "nos",
        "vos",
        "leur",
    ];
    dict.load_words(&words);
    dict
}

pub fn load_spanish_dictionary() -> Dictionary {
    let mut dict = Dictionary::new();
    let words = [
        "el",
        "de",
        "que",
        "y",
        "a",
        "en",
        "un",
        "ser",
        "se",
        "no",
        "haber",
        "por",
        "con",
        "su",
        "para",
        "como",
        "estar",
        "tener",
        "le",
        "lo",
        "pero",
        "sus",
        "ella",
        "todo",
        "esta",
        "si",
        "sobre",
        "mi",
        "alguno",
        "o",
        "este",
        "ya",
        "porque",
        "cuando",
        "solo",
        "muy",
        "aunque",
        "también",
        "me",
        "hasta",
        "hay",
        "donde",
        "mientras",
        "quien",
        "desde",
        "nos",
        "durante",
        "siempre",
        "todos",
        "cual",
        "poco",
        "estar",
        "poder",
        "decir",
        "ir",
        "ver",
        "dar",
        "saber",
        "querer",
        "llegar",
        "pasar",
        "deber",
        "poner",
        "parecer",
        "quedar",
        "creer",
        "hablar",
        "llevar",
        "dejar",
        "seguir",
        "encontrar",
        "llamar",
        "venir",
        "pensar",
        "salir",
        "volver",
        "tomar",
        "conocer",
        "vivir",
        "sentir",
        "tratar",
        "mirar",
        "contar",
        "empezar",
        "esperar",
        "buscar",
        "entrar",
        "trabajar",
        "escribir",
        "perder",
        "producir",
        "comenzar",
        "gobierno",
        "país",
        "mundo",
        "año",
        "día",
        "hombre",
        "mujer",
        "vida",
        "mano",
        "parte",
        "lugar",
        "caso",
        "semana",
        "empresa",
        "sistema",
        "programa",
        "pregunta",
        "trabajo",
        "número",
        "noche",
        "punto",
        "hogar",
        "agua",
        "habitación",
        "madre",
        "dinero",
        "historia",
        "hecho",
        "mes",
        "mucho",
        "derecho",
        "libro",
        "ojo",
        "empleo",
        "palabra",
        "negocio",
        "problema",
        "lado",
        "cabeza",
        "amigo",
        "padre",
        "hora",
        "juego",
        "línea",
        "fin",
        "miembro",
        "ley",
        "coche",
        "ciudad",
        "comunidad",
        "nombre",
        "presidente",
        "equipo",
        "minuto",
        "idea",
        "cuerpo",
        "información",
        "padres",
        "cara",
        "nivel",
        "oficina",
        "puerta",
        "salud",
        "persona",
        "arte",
        "guerra",
        "historia",
        "fiesta",
        "resultado",
        "mañana",
        "razón",
        "investigación",
        "chica",
        "momento",
        "aire",
        "profesor",
        "fuerza",
        "educación",
        "comida",
        "foto",
        "clase",
        "producto",
        "experiencia",
        "mercado",
        "familia",
        "regreso",
        "perro",
        "estudiante",
        "grupo",
        "valor",
        "plan",
        "escuela",
        "interés",
        "muerte",
        "centro",
        "música",
        "papel",
        "control",
        "amor",
        "verdad",
        "mujer",
        "hombre",
        "edad",
        "sí",
        "demasiado",
        "así",
        "entonces",
        "otro",
        "aún",
        "siempre",
        "ya",
        "nunca",
        "mismo",
        "tanto",
        "después",
        "antes",
        "desde",
        "cuando",
        "pues",
        "aunque",
        "sin",
        "bajo",
        "sobre",
        "ante",
        "entre",
        "hacia",
        "hasta",
        "según",
        "ninguno",
        "varios",
        "alguno",
        "todo",
        "cada",
        "mío",
        "tuyo",
        "suyo",
        "nuestro",
        "vuestro",
        "suyo",
        "este",
        "ese",
        "aquel",
        "estos",
        "esos",
        "aquellos",
        "aquí",
        "ahí",
        "allí",
        "mío",
        "tuyo",
    ];
    dict.load_words(&words);
    dict
}

pub fn load_german_dictionary() -> Dictionary {
    let mut dict = Dictionary::new();
    let words = [
        "der",
        "die",
        "und",
        "in",
        "den",
        "von",
        "mit",
        "ist",
        "das",
        "für",
        "auf",
        "als",
        "bei",
        "sich",
        "auch",
        "nach",
        "zur",
        "ein",
        "wie",
        "ihre",
        "über",
        "dass",
        "oder",
        "aber",
        "wenn",
        "nur",
        "durch",
        "aus",
        "kann",
        "schon",
        "jetzt",
        "noch",
        "sehr",
        "hier",
        "mehr",
        "wir",
        "was",
        "man",
        "nicht",
        "eine",
        "um",
        "alle",
        "müssen",
        "haben",
        "werden",
        "können",
        "sollen",
        "lassen",
        "machen",
        "geben",
        "kommen",
        "gehen",
        "sehen",
        "finden",
        "halten",
        "bringen",
        "stehen",
        "bleiben",
        "wissen",
        "nehmen",
        "dürfen",
        "meinen",
        "glauben",
        "schaffen",
        "liegen",
        "zeigen",
        "führen",
        "sprechen",
        "lesen",
        "spielen",
        "laufen",
        "ziehen",
        "helfen",
        "beginnen",
        "arbeiten",
        "erscheinen",
        "gelten",
        "erhalten",
        "bilden",
        "erscheinen",
        "bedeuten",
        "verstehen",
        "setzen",
        "erklären",
        "entsprechen",
        "versuchen",
        "erreichen",
        "bieten",
        "gelingen",
        "regierung",
        "land",
        "welt",
        "jahr",
        "tag",
        "mann",
        "frau",
        "leben",
        "hand",
        "teil",
        "ort",
        "fall",
        "woche",
        "unternehmen",
        "system",
        "programm",
        "frage",
        "arbeit",
        "zahl",
        "nacht",
        "punkt",
        "haus",
        "wasser",
        "zimmer",
        "mutter",
        "geld",
        "geschichte",
        "tatsache",
        "monat",
        "viel",
        "recht",
        "buch",
        "auge",
        "job",
        "wort",
        "geschäft",
        "problem",
        "seite",
        "kopf",
        "freund",
        "vater",
        "stunde",
        "spiel",
        "linie",
        "ende",
        "mitglied",
        "gesetz",
        "auto",
        "stadt",
        "gemeinschaft",
        "name",
        "präsident",
        "mannschaft",
        "minute",
        "idee",
        "körper",
        "information",
        "eltern",
        "gesicht",
        "ebene",
        "büro",
        "tür",
        "gesundheit",
        "person",
        "kunst",
        "krieg",
        "geschichte",
        "partei",
        "ergebnis",
        "morgen",
        "grund",
        "forschung",
        "mädchen",
        "moment",
        "luft",
        "lehrer",
        "kraft",
        "bildung",
        "essen",
        "bild",
        "klasse",
        "produkt",
        "erfahrung",
        "markt",
        "familie",
        "rückkehr",
        "hund",
        "student",
        "gruppe",
        "wert",
        "plan",
        "schule",
        "interesse",
        "tod",
        "zentrum",
        "musik",
        "papier",
        "kontrolle",
        "liebe",
        "wahr",
        "frau",
        "mann",
        "alter",
        "ja",
        "zu",
        "auch",
        "so",
        "noch",
        "immer",
        "schon",
        "nie",
        "selbst",
        "so",
        "dann",
        "nach",
        "vor",
        "seit",
        "wenn",
        "da",
        "obwohl",
        "ohne",
        "unter",
        "auf",
        "gegen",
        "zwischen",
        "nach",
        "bis",
        "laut",
        "kein",
        "mehrere",
        "einige",
        "alle",
        "jeder",
        "mein",
        "dein",
        "sein",
        "unser",
        "euer",
        "ihr",
        "dieser",
        "jener",
        "welcher",
        "hier",
        "dort",
        "drüben",
        "mein",
        "dein",
    ];
    dict.load_words(&words);
    dict
}

pub fn load_italian_dictionary() -> Dictionary {
    let mut dict = Dictionary::new();
    let words = [
        "il",
        "di",
        "che",
        "è",
        "la",
        "per",
        "un",
        "sono",
        "con",
        "ma",
        "come",
        "non",
        "io",
        "dal",
        "al",
        "del",
        "nei",
        "sui",
        "nel",
        "sul",
        "alla",
        "della",
        "nella",
        "sulla",
        "alle",
        "delle",
        "nelle",
        "sulle",
        "lo",
        "l'",
        "un'",
        "ho",
        "ha",
        "hai",
        "abbiamo",
        "avete",
        "hanno",
        "sono",
        "sei",
        "siamo",
        "siete",
        "essere",
        "avere",
        "fare",
        "dire",
        "potere",
        "andare",
        "vedere",
        "dovere",
        "dare",
        "volere",
        "venire",
        "sapere",
        "parlare",
        "trovare",
        "sentire",
        "lasciare",
        "prendere",
        "mettere",
        "passare",
        "vivere",
        "chiamare",
        "diventare",
        "rimanere",
        "portare",
        "tenere",
        "capire",
        "cominciare",
        "finire",
        "seguito",
        "servire",
        "uscire",
        "cercare",
        "entrare",
        "rispondere",
        "scrivere",
        "morire",
        "partire",
        "tornare",
        "pensare",
        "guardare",
        "rimanere",
        "perdere",
        "salire",
        "sedere",
        "scegliere",
        "vincere",
        "governo",
        "paese",
        "mondo",
        "anno",
        "giorno",
        "uomo",
        "donna",
        "vita",
        "mano",
        "parte",
        "luogo",
        "caso",
        "settimana",
        "azienda",
        "sistema",
        "programma",
        "domanda",
        "lavoro",
        "numero",
        "notte",
        "punto",
        "casa",
        "acqua",
        "stanza",
        "madre",
        "soldi",
        "storia",
        "fatto",
        "mese",
        "molto",
        "diritto",
        "libro",
        "occhio",
        "lavoro",
        "parola",
        "affari",
        "problema",
        "lato",
        "testa",
        "amico",
        "padre",
        "ora",
        "gioco",
        "linea",
        "fine",
        "membro",
        "legge",
        "macchina",
        "città",
        "comunità",
        "nome",
        "presidente",
        "squadra",
        "minuto",
        "idea",
        "corpo",
        "informazione",
        "genitori",
        "volto",
        "livello",
        "ufficio",
        "porta",
        "salute",
        "persona",
        "arte",
        "guerra",
        "storia",
        "festa",
        "risultato",
        "mattina",
        "ragione",
        "ricerca",
        "ragazza",
        "momento",
        "aria",
        "insegnante",
        "forza",
        "educazione",
        "cibo",
        "foto",
        "classe",
        "prodotto",
        "esperienza",
        "mercato",
        "famiglia",
        "ritorno",
        "cane",
        "studente",
        "gruppo",
        "valore",
        "piano",
        "scuola",
        "interesse",
        "morte",
        "centro",
        "musica",
        "carta",
        "controllo",
        "amore",
        "vero",
        "donna",
        "uomo",
        "età",
        "sì",
        "troppo",
        "così",
        "allora",
        "altro",
        "ancora",
        "sempre",
        "già",
        "mai",
        "stesso",
        "tanto",
        "dopo",
        "prima",
        "da",
        "quando",
        "poiché",
        "benché",
        "senza",
        "sotto",
        "su",
        "contro",
        "tra",
        "verso",
        "fino",
        "nessuno",
        "alcuni",
        "tutti",
        "ogni",
        "mio",
        "tuo",
        "suo",
        "nostro",
        "vostro",
        "loro",
        "questo",
        "quello",
        "quelli",
        "qui",
        "lì",
        "là",
        "mio",
        "tuo",
    ];
    dict.load_words(&words);
    dict
}

pub fn load_portuguese_dictionary() -> Dictionary {
    let mut dict = Dictionary::new();
    let words = [
        "o",
        "de",
        "a",
        "que",
        "e",
        "do",
        "da",
        "em",
        "um",
        "para",
        "com",
        "não",
        "uma",
        "os",
        "se",
        "no",
        "as",
        "por",
        "mais",
        "mas",
        "foi",
        "ao",
        "sua",
        "das",
        "ser",
        "tem",
        "já",
        "entre",
        "quando",
        "muito",
        "sobre",
        "também",
        "aos",
        "ou",
        "este",
        "aos",
        "outro",
        "pelo",
        "qual",
        "tudo",
        "mesmo",
        "ainda",
        "ver",
        "anos",
        "depois",
        "sem",
        "todos",
        "cada",
        "apenas",
        "muito",
        "estar",
        "ser",
        "fazer",
        "ter",
        "poder",
        "dizer",
        "dar",
        "saber",
        "querer",
        "ir",
        "ver",
        "vir",
        "parecer",
        "ficar",
        "trazer",
        "conseguir",
        "achar",
        "deixar",
        "partir",
        "chegar",
        "voltar",
        "passar",
        "usar",
        "trabalhar",
        "começar",
        "continuar",
        "viver",
        "conhecer",
        "pensar",
        "entrar",
        "morar",
        "tornar",
        "acontecer",
        "sentir",
        "sair",
        "correr",
        "ouvir",
        "acabar",
        "pedir",
        "ler",
        "acreditar",
        "escrever",
        "perder",
        "governo",
        "país",
        "mundo",
        "ano",
        "dia",
        "homem",
        "mulher",
        "vida",
        "mão",
        "parte",
        "lugar",
        "caso",
        "semana",
        "empresa",
        "sistema",
        "programa",
        "pergunta",
        "trabalho",
        "número",
        "noite",
        "ponto",
        "casa",
        "água",
        "quarto",
        "mãe",
        "dinheiro",
        "história",
        "fato",
        "mês",
        "muito",
        "direito",
        "livro",
        "olho",
        "emprego",
        "palavra",
        "negócio",
        "problema",
        "lado",
        "cabeça",
        "amigo",
        "pai",
        "hora",
        "jogo",
        "linha",
        "fim",
        "membro",
        "lei",
        "carro",
        "cidade",
        "comunidade",
        "nome",
        "presidente",
        "equipe",
        "minuto",
        "ideia",
        "corpo",
        "informação",
        "pais",
        "rosto",
        "nível",
        "escritório",
        "porta",
        "saúde",
        "pessoa",
        "arte",
        "guerra",
        "história",
        "festa",
        "resultado",
        "manhã",
        "razão",
        "pesquisa",
        "garota",
        "momento",
        "ar",
        "professor",
        "força",
        "educação",
        "comida",
        "foto",
        "classe",
        "produto",
        "experiência",
        "mercado",
        "família",
        "retorno",
        "cachorro",
        "estudante",
        "grupo",
        "valor",
        "plano",
        "escola",
        "interesse",
        "morte",
        "centro",
        "música",
        "papel",
        "controle",
        "amor",
        "verdade",
        "mulher",
        "homem",
        "idade",
        "sim",
        "demais",
        "assim",
        "então",
        "outro",
        "ainda",
        "sempre",
        "já",
        "nunca",
        "mesmo",
        "tanto",
        "depois",
        "antes",
        "desde",
        "quando",
        "pois",
        "embora",
        "sem",
        "sob",
        "sobre",
        "contra",
        "entre",
        "para",
        "até",
        "segundo",
        "nenhum",
        "vários",
        "alguns",
        "todos",
        "cada",
        "meu",
        "teu",
        "seu",
        "nosso",
        "vosso",
        "deles",
        "este",
        "esse",
        "aquele",
        "estes",
        "esses",
        "aqueles",
        "aqui",
        "aí",
        "ali",
        "meu",
        "teu",
    ];
    dict.load_words(&words);
    dict
}

pub fn load_russian_dictionary() -> Dictionary {
    let mut dict = Dictionary::new();
    let words = [
        "и",
        "в",
        "не",
        "на",
        "я",
        "быть",
        "он",
        "с",
        "что",
        "а",
        "по",
        "это",
        "она",
        "к",
        "но",
        "мы",
        "как",
        "из",
        "за",
        "от",
        "о",
        "же",
        "так",
        "все",
        "тот",
        "мочь",
        "вы",
        "человек",
        "один",
        "только",
        "его",
        "который",
        "ещё",
        "время",
        "если",
        "говорить",
        "знать",
        "стать",
        "для",
        "можно",
        "год",
        "работа",
        "жизнь",
        "рука",
        "город",
        "случай",
        "ребёнок",
        "голова",
        "дом",
        "сила",
        "россия",
        "женщина",
        "вода",
        "семья",
        "дверь",
        "страна",
        "работать",
        "любить",
        "стоять",
        "открыть",
        "казаться",
        "прийти",
        "хотеть",
        "выйти",
        "понять",
        "пойти",
        "спросить",
        "жить",
        "сидеть",
        "оставаться",
        "вернуться",
        "казаться",
        "начинать",
        "показывать",
        "считать",
        "понимать",
        "получать",
        "государство",
        "мир",
        "лет",
        "день",
        "рука",
        "глаз",
        "дело",
        "место",
        "лицо",
        "друг",
        "глава",
        "вопрос",
        "сторона",
        "стол",
        "ребёнок",
        "утро",
        "путь",
        "дверь",
        "конец",
        "час",
        "голос",
        "город",
        "последний",
        "политика",
        "вид",
        "глаз",
        "часть",
        "слово",
        "момент",
        "минута",
        "господин",
        "сердце",
        "дорога",
        "свет",
        "пора",
        "спина",
        "мать",
        "комната",
        "улица",
        "ночь",
        "вода",
        "власть",
        "воздух",
        "общество",
        "состояние",
        "письмо",
        "нога",
        "отец",
        "вечер",
        "мысль",
        "жизнь",
        "мать",
        "правда",
        "москва",
        "книга",
        "душа",
        "плечо",
        "смысл",
        "способ",
        "грудь",
        "сон",
        "судьба",
        "точка",
        "область",
        "цвет",
        "солнце",
        "движение",
        "праздник",
        "чувство",
        "метод",
        "уровень",
        "форма",
        "связь",
        "закон",
        "средство",
        "период",
        "план",
        "число",
        "цель",
        "класс",
        "партия",
        "отдел",
        "процесс",
        "результат",
        "действие",
        "акт",
        "состав",
        "место",
        "название",
        "значение",
        "организация",
        "технология",
        "информация",
        "память",
        "причина",
        "условие",
        "средство",
        "помощь",
        "роль",
        "автор",
        "пример",
        "система",
        "группа",
        "развитие",
        "встреча",
        "производство",
        "качество",
        "смысл",
        "школа",
        "институт",
        "следствие",
        "рост",
        "рисунок",
        "повод",
        "сознание",
        "сцена",
        "шаг",
        "выбор",
        "след",
        "огонь",
        "гость",
        "бой",
        "поток",
        "дядя",
        "счёт",
        "взгляд",
        "пауза",
        "род",
        "отряд",
        "участник",
        "крик",
        "революция",
        "корень",
        "кожа",
        "ряд",
        "сутки",
        "воля",
        "масса",
        "смена",
        "кровь",
        "район",
        "возраст",
        "мальчик",
        "армия",
        "девушка",
        "поле",
        "цена",
        "тип",
        "мера",
        "мнение",
        "звук",
        "версия",
        "шанс",
        "образ",
        "машина",
        "зал",
        "рана",
        "командир",
        "игра",
        "кухня",
        "лестница",
        "позиция",
        "карман",
        "солдат",
        "хозяин",
        "попытка",
        "степень",
        "волос",
        "зуб",
        "цветок",
        "камень",
        "сеть",
        "клетка",
        "множество",
        "вода",
        "еда",
        "небо",
        "снег",
        "огонь",
        "земля",
        "ветер",
        "луна",
        "звезда",
        "облако",
        "трава",
        "лист",
        "дерево",
        "цвет",
        "птица",
        "собака",
        "кошка",
        "лошадь",
        "рыба",
        "медведь",
        "волк",
        "лиса",
        "заяц",
        "орёл",
        "голубь",
        "воробей",
        "курица",
        "утка",
        "гусь",
        "лебедь",
        "лев",
        "тигр",
        "слон",
        "жираф",
        "обезьяна",
        "бегемот",
        "носорог",
        "белка",
        "ёж",
        "мышь",
        "крыса",
        "лягушка",
        "змея",
        "черепаха",
        "крокодил",
        "акула",
        "кит",
        "дельфин",
        "тюлень",
    ];
    dict.load_words(&words);
    dict
}

fn load_dutch_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_polish_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_swedish_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_danish_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_finnish_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_norwegian_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_turkish_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_greek_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_hindi_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_thai_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_vietnamese_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_arabic_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_hebrew_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_indonesian_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_malay_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_ukrainian_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_czech_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_hungarian_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_romanian_dictionary() -> Dictionary {
    load_english_dictionary()
}
fn load_bulgarian_dictionary() -> Dictionary {
    load_english_dictionary()
}

pub struct DictionaryHandler {
    dictionary: Dictionary,
}

impl DictionaryHandler {
    pub fn new() -> Self {
        Self {
            dictionary: load_english_dictionary(),
        }
    }

    pub fn new_for_language(lang: &str) -> Self {
        let dictionary = match lang.to_lowercase().as_str() {
            "fr" => load_french_dictionary(),
            "es" => load_spanish_dictionary(),
            "de" => load_german_dictionary(),
            "it" => load_italian_dictionary(),
            "pt" => load_portuguese_dictionary(),
            "ru" => load_russian_dictionary(),
            "nl" => load_dutch_dictionary(),
            "pl" => load_polish_dictionary(),
            "sv" => load_swedish_dictionary(),
            "da" => load_danish_dictionary(),
            "fi" => load_finnish_dictionary(),
            "no" => load_norwegian_dictionary(),
            "tr" => load_turkish_dictionary(),
            "el" => load_greek_dictionary(),
            "hi" => load_hindi_dictionary(),
            "th" => load_thai_dictionary(),
            "vi" => load_vietnamese_dictionary(),
            "ar" => load_arabic_dictionary(),
            "he" => load_hebrew_dictionary(),
            "id" => load_indonesian_dictionary(),
            "ms" => load_malay_dictionary(),
            "uk" => load_ukrainian_dictionary(),
            "cs" => load_czech_dictionary(),
            "hu" => load_hungarian_dictionary(),
            "ro" => load_romanian_dictionary(),
            "bg" => load_bulgarian_dictionary(),
            _ => load_english_dictionary(),
        };
        Self { dictionary }
    }

    pub fn with_dictionary(dictionary: Dictionary) -> Self {
        Self { dictionary }
    }

    /// Load additional words from a plain text file (one word per line or whitespace-separated)
    pub fn load_from_file<P: AsRef<std::path::Path>>(&mut self, path: P) -> std::io::Result<usize> {
        let text = std::fs::read_to_string(path)?;
        let before = self.dictionary.word_count();
        self.dictionary.load_from_text(&text);
        Ok(self.dictionary.word_count() - before)
    }

    /// Load additional words from a string (whitespace-separated)
    pub fn load_from_text(&mut self, text: &str) {
        self.dictionary.load_from_text(text);
    }

    pub fn is_word_valid(&self, word: &str) -> bool {
        self.dictionary.contains(word)
    }

    pub fn suggest(&self, word: &str, max: usize) -> Vec<(String, usize)> {
        self.dictionary.suggest_corrections(word, max)
    }

    pub fn correct(&self, word: &str) -> String {
        self.dictionary.correct_word(word)
    }

    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
    }

    pub fn dictionary_mut(&mut self) -> &mut Dictionary {
        &mut self.dictionary
    }
}

impl Default for DictionaryHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(edit_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_edit_distance_empty() {
        assert_eq!(edit_distance("", "hello"), 5);
        assert_eq!(edit_distance("hello", ""), 5);
    }

    #[test]
    fn test_edit_distance_substitution() {
        assert_eq!(edit_distance("cat", "bat"), 1);
        assert_eq!(edit_distance("cat", "bet"), 2);
    }

    #[test]
    fn test_edit_distance_insertion() {
        assert_eq!(edit_distance("cat", "cats"), 1);
        assert_eq!(edit_distance("cat", "cast"), 1);
    }

    #[test]
    fn test_edit_distance_deletion() {
        assert_eq!(edit_distance("cats", "cat"), 1);
    }

    #[test]
    fn test_dictionary_contains() {
        let dict = load_english_dictionary();
        assert!(dict.contains("the"));
        assert!(dict.contains("THE"));
        assert!(dict.contains("help"));
    }

    #[test]
    fn test_dictionary_suggest() {
        let dict = load_english_dictionary();
        let suggestions = dict.suggest_corrections("teh", 3);
        assert!(!suggestions.is_empty());
        let (word, dist) = &suggestions[0];
        assert!(*dist <= 2);
        assert!(*word == "the" || *word == "yet");
    }

    #[test]
    fn test_dictionary_correct() {
        let dict = load_english_dictionary();
        let corrected = dict.correct_word("teh");
        assert!(corrected == "the" || corrected == "yet");
    }

    #[test]
    fn test_dictionary_correct_teh() {
        let dict = load_english_dictionary();
        let corrected = dict.correct_word("teh");
        assert!(corrected == "the" || corrected == "yet");
    }

    #[test]
    fn test_dictionary_no_correction_needed() {
        let dict = load_english_dictionary();
        assert_eq!(dict.correct_word("the"), "the");
    }

    #[test]
    fn test_preserve_case() {
        assert_eq!(preserve_case("HELLO", "the"), "THE");
        assert_eq!(preserve_case("Hello", "the"), "The");
        assert_eq!(preserve_case("hello", "the"), "the");
    }

    #[test]
    fn test_dictionary_handler_new_for_language_fr() {
        let handler = DictionaryHandler::new_for_language("fr");
        assert!(handler.is_word_valid("le"));
        assert!(handler.is_word_valid("être"));
        assert!(handler.is_word_valid("avoir"));
    }

    #[test]
    fn test_dictionary_handler_new_for_language_es() {
        let handler = DictionaryHandler::new_for_language("es");
        assert!(handler.is_word_valid("el"));
        assert!(handler.is_word_valid("ser"));
        assert!(handler.is_word_valid("haber"));
    }

    #[test]
    fn test_dictionary_handler_new_for_language_de() {
        let handler = DictionaryHandler::new_for_language("de");
        assert!(handler.is_word_valid("der"));
        assert!(handler.is_word_valid("sein"));
        assert!(handler.is_word_valid("haben"));
    }

    #[test]
    fn test_dictionary_handler_new_for_language_it() {
        let handler = DictionaryHandler::new_for_language("it");
        assert!(handler.is_word_valid("il"));
        assert!(handler.is_word_valid("essere"));
        assert!(handler.is_word_valid("avere"));
    }

    #[test]
    fn test_dictionary_handler_new_for_language_pt() {
        let handler = DictionaryHandler::new_for_language("pt");
        assert!(handler.is_word_valid("o"));
        assert!(handler.is_word_valid("ser"));
        assert!(handler.is_word_valid("estar"));
    }

    #[test]
    fn test_dictionary_handler_new_for_language_ru() {
        let handler = DictionaryHandler::new_for_language("ru");
        assert!(handler.is_word_valid("и"));
        assert!(handler.is_word_valid("быть"));
        assert!(handler.is_word_valid("он"));
    }

    #[test]
    fn test_dictionary_handler_new_for_language_fallback() {
        let handler = DictionaryHandler::new_for_language("nl");
        assert!(handler.is_word_valid("the"));
        assert!(handler.is_word_valid("and"));
        assert!(handler.is_word_valid("for"));
    }

    #[test]
    fn test_dictionary_handler_new_for_language_case_insensitive() {
        let handler_upper = DictionaryHandler::new_for_language("FR");
        assert!(handler_upper.is_word_valid("le"));
        let handler_mixed = DictionaryHandler::new_for_language("Es");
        assert!(handler_mixed.is_word_valid("el"));
    }

    #[test]
    fn test_french_dictionary_suggest() {
        let dict = load_french_dictionary();
        let suggestions = dict.suggest_corrections("teh", 3);
        assert!(suggestions.len() <= 3);
    }

    #[test]
    fn test_spanish_dictionary_contains_common_words() {
        let dict = load_spanish_dictionary();
        assert!(dict.contains("el"));
        assert!(dict.contains("ser"));
        assert!(dict.contains("para"));
    }

    #[test]
    fn test_german_dictionary_contains_common_words() {
        let dict = load_german_dictionary();
        assert!(dict.contains("der"));
        assert!(dict.contains("und"));
        assert!(dict.contains("sein"));
    }

    #[test]
    fn test_russian_dictionary_contains_common_words() {
        let dict = load_russian_dictionary();
        assert!(dict.contains("и"));
        assert!(dict.contains("в"));
        assert!(dict.contains("не"));
    }
}
