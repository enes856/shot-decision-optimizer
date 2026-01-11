# Report: Shot Decision Optimizer

---

## 1. Problemstellung, Ansatz und Lösung

### 1.1 Forschungsziel

Die traditionelle Basketball-Analyse stützt sich stark auf deskriptive Statistiken und post hoc Visualisierungen wie Heatmaps, die zwar Leistungs Hotspots aufzeigen, jedoch keine prädiktive, kontextsensitive Entscheidungshilfe in Echtzeit bieten. Die zentrale Problemstellung dieses Projekts war die Konzeption und Implementierung eines **Shot Decision Optimizer**; eines datengestützten Systems zur Quantifizierung der Wurfqualität in der NBA.

Das Forschungsziel war es, von einer reinen Vergangenheitsbetrachtung zu einer prädiktiven Analyse überzugehen. Anstelle der Frage „Woher trifft ein Spieler?“ sollte die Frage „Wie hoch ist die Erfolgswahrscheinlichkeit eines Wurfes unter Berücksichtigung aller relevanten kontextuellen Faktoren?“ beantwortet werden. Hierzu wurde die Metrik **xPts (Expected Points)** entwickelt, die das Produkt aus der vorhergesagten Trefferwahrscheinlichkeit und dem Punktewert des Wurfes darstellt:

$xPts = P(\text{Make}) \times \text{Shot Value}$

Dieses System ermöglicht eine objektive Bewertung der Entscheidungsfindung von Spielern und bildet die Grundlage für strategische Optimierungen im Offensivspiel.

### 1.2 Von Rohdaten zur Inferenz

Die entwickelte Pipeline folgt einem stringenten logischen Aufbau, der den „Roten Faden“ von der Datenakquise bis zur finalen Modellinferenz bildet:

1.  **Datenakquise und -speicherung:** Mittels der `nba_api` wurden Rohdaten der `ShotChartDetail` und `PlayByPlayV3` Endpunkte für mehrere Saisons extrahiert. Die Implementierung in `utils/load_data_utils.py` kapselt diese Logik und speichert die Daten effizient im **Parquet-Format**, um schnelle Lesezugriffe für die nachfolgenden Analyseschritte zu gewährleisten.

2.  **Explorative Datenanalyse (EDA):** Das Skript `scripts/eda.py` (alternativ unter: `notebooks/eda.ipynb`) nutzt Funktionen aus `utils/eda_utils.py`, um ein tiefes Verständnis der Daten zu erlangen. Analysiert wurden Verteilungen (z.B. Wurfdistanz), Korrelationen (z.B. Wurfeffizienz vs. verbleibende Zeit) und Trends über Saisons hinweg. Diese Phase war entscheidend, um Hypothesen für das Feature Engineering zu validieren.

3.  **Feature Engineering:** Dies war der Kern der Wertschöpfung. In `utils/feature_engineering_utils.py` wurden Features implementiert, die sich in vier Kategorien einteilen lassen:
    *   **Geometrische Features:** `SHOT_ANGLE`, `CUSTOM_SHOT_ZONE`
    *   **Kontext-Features:** `SHOTCLOCK` (aufwändig aus Play-by-Play-Daten rekonstruiert), `TIME_REMAINING`
    *   **Historische Spieler-Features:** `PLAYER_SEASON_FG_PCT`, `PLAYER_ZONE_FG_PCT`, `LAST_5_SHOTS_MADE` (Momentum), die über `rolling()` und `shift()` leakagefrei berechnet wurden.
    *   **Gegner-Features:** `OPP_DEF_STRENGTH`

4.  **Modellierung und Training:** In `utils/logreg_modeling_utils.py` und `utils/xg_boost_modeling_utils.py` wurden zwei komplementäre Modellierungsansätze implementiert: **Logistische Regression** für Interpretierbarkeit und **XGBoost** für die Erfassung nicht linearer Zusammenhänge. Die Trainingspipeline, orchestriert in `scripts/xg_boost_modeling.py`, umfasste ein chronologisches Train Test Split nach Saisons, um die Vorhersage auf zukünftigen Daten realistisch zu simulieren.

5.  **Evaluation und Inferenz:** Die Modelle wurden nicht nur anhand klassischer Metriken (AUC-ROC, Log-Loss), sondern auch über spiel aggregierte `xPts`-Werte evaluiert. Diese Makro Validierung prüft, ob die Summe der vorhergesagten `xPts` mit dem tatsächlichen Spielergebnis korreliert, was für die praktische Relevanz des Modells entscheidend ist.


### 1.3 Modulare Architektur

Die strikte Trennung von Logik in den `utils`-Modulen und deren Orchestrierung in den `scripts`-Notebooks war ein fundamentaler Architekturansatz. Diese Modularität ermöglichte eine effiziente, parallele Entwicklung:

*   **`eda_utils.py`:** Standardisierte Analysefunktionen, die teamübergreifend für schnelle Datenvalidierung genutzt werden konnten.
*   **`feature_engineering_utils.py`:** Komplexe Berechnungen (z.B. `calculate_shot_clock_from_pbp`) wurden als eigenständige, testbare Einheiten gekapselt.
*   **`logreg_modeling_utils.py` & `xg_boost_modeling_utils.py`:** Die gesamte Pipeline von Preprocessing über Training bis zur Evaluation wurde als wiederverwendbare Funktion (z.B. `build_pipeline`, `evaluate_model`) strukturiert.


---

## 2. Technische Dokumentation

### 2.1 Dokumentation Daten

#### 2.1.1 Datenprovenienz und -schema

Die Datengrundlage des Projekts bilden zwei Endpunkte der offiziellen **NBA Stats API**:

1.  **ShotChartDetail:** Liefert detaillierte Informationen zu jedem Wurfversuch, inklusive XY-Koordinaten, Spieler-IDs und deskriptiven Metadaten.
2.  **PlayByPlayV3:** Stellt einen chronologischen Log aller Spielereignisse bereit, der zur Rekonstruktion dynamischer Kontextfeatures wie der Shot Clock unerlässlich war.

Die Daten wurden für die Saisons 2016-17 bis 2024-25 extrahiert und umfassen mehrere Millionen Wurfereignisse.

**Architektonische Entscheidung zur Datenspeicherung:**
Die Rohdaten sowie die prozessierten, mit Features angereicherten Daten wurden im **Apache Parquet-Format** gespeichert. Diese Entscheidung wurde bewusst getroffen, um Big-Data-Anforderungen zu adressieren:
*   **Effizienz:** Parquet ist ein spaltenbasiertes Speicherformat. Analytische Abfragen, die typischerweise nur eine Teilmenge der Spalten benötigen (z.B. `GROUP BY PLAYER_ID`), können Datenblöcke überspringen und sind dadurch signifikant performanter als bei zeilenbasierten Formaten wie CSV.
*   **Kompression:** Parquet bietet eine hocheffiziente Kompression, was die Speicherkosten und I/O-Latenz reduziert.
*   **Schema-Evolution:** Das Format unterstützt die Weiterentwicklung des Datenschemas, was für iterative Feature-Engineering-Zyklen vorteilhaft ist.

Die Verzeichnisstruktur `data/raw`, `data/predict` und  `data/processed` sorgt für eine klare Trennung der Datenintegritätsstufen und unterstützt reproduzierbare Analyse-Pipelines.


#### 2.1.2 Pre-processing Pipeline: Die Logik hinter den Transformationen

Das Feature Engineering war der entscheidende Schritt, um aus rohen Koordinaten und Zeitstempeln prädiktiv wertvolle Informationen zu generieren.

**Normalisierung und Skalierung:**
Die numerischen Features wurden mittels `StandardScaler` standardisiert. Dies ist eine Voraussetzung für Modelle wie die Logistische Regression, deren Konvergenz von einer normalisierten Feature Landschaft abhängt. Die Formel für die Z-Score-Normalisierung lautet:

$z = \frac{x - \mu}{\sigma}$

wobei $\mu$ der Mittelwert und $\sigma$ die Standardabweichung des Features ist.

**Dynamische, leakagefreie Features:**
Die Berechnung historischer Spielerstatistiken (`PLAYER_SEASON_FG_PCT`, `PLAYER_ZONE_FG_PCT`) erforderte besondere Sorgfalt, um **Data Leakage** zu vermeiden. Ein Wurf darf nur auf Basis von Informationen bewertet werden, die *vor* diesem Wurf verfügbar waren. Dies wurde durch eine Kombination aus `groupby()`, `shift(1)` und `rolling()` in `pandas` realisiert. `shift(1) ` stellt sicher, dass der aktuelle Wurf nicht in seine eigene Prädiktorgenerierung einfließt.

**Rekonstruktion der Shot Clock:**
Das Feature `SHOTCLOCK` war in den Rohdaten nicht direkt verfügbar. Die Funktion `calculate_shot_clock_from_pbp` in `feature_engineering_utils.py` implementiert eine komplexe Logik zur Approximation:
1.  **Besitzwechselerkennung:** Ein Besitzwechsel wird identifiziert, wenn sich die `teamId` im Play-by-Play-Log ändert.
2.  **Zeitmessung:** Die Zeitdifferenz zwischen dem Beginn des Ballbesitzes und dem Wurfereignis wird berechnet.
3.  **Approximation:** Die verbleibende Shot Clock wird als $24s - \Delta t$ berechnet und auf plausible Werte (`[0, 24]`) clippt.

Dieses Feature ist ein entscheidender Prädiktor für die Wurfqualität, da es den Handlungsdruck auf den Werfer quantifiziert.

### 2.2 Dokumentation Modell

#### 2.2.1 Mathematische Fundierung der Modelle

Es wurden bewusst zwei Modellfamilien gewählt, um den Trade-off zwischen Interpretierbarkeit und Prädiktionskraft zu analysieren.

**Logistische Regression:**
Dieses Modell schätzt die Wahrscheinlichkeit eines Wurferfolgs $P(Y=1|X)$ über die logistische Funktion. Die zu minimierende Zielfunktion ist die **binäre Kreuzentropie (Log-Loss)**, die die Abweichung zwischen vorhergesagter Wahrscheinlichkeit $\hat{p}_i$ und tatsächlichem Ergebnis $y_i$ bestraft:

$L_{\text{Log-Loss}} = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i)]$

Die Stärke des Modells liegt in seinen interpretierbaren Koeffizienten, die den Einfluss jedes Features direkt quantifizieren.

**XGBoost (Extreme Gradient Boosting):**
XGBoost ist ein Ensemble-Verfahren, das sequenziell schwache Lerner (Entscheidungsbäume) trainiert, wobei jeder neue Baum die Residuen (Fehler) der vorherigen Bäume korrigiert. Die Zielfunktion ist komplexer und kombiniert eine Verlustfunktion (z.B. Log-Loss) mit einem Regularisierungsterm $\Omega$, der die Komplexität der Bäume bestraft und so Overfitting verhindert:

$\text{Obj}(t) = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$

Die Stärke von XGBoost liegt in seiner Fähigkeit, hochdimensionale, nicht-lineare Interaktionen zwischen Features zu modellieren.

#### 2.2.2 Evaluationsmetriken und Interpretation der Ergebnisse

Die Modellevaluation war mehrdimensional und fokussierte sich auf die für den Anwendungsfall relevanten Aspekte.

**Kalibrierung (Brier Score):**
Die Zuverlässigkeit der `xPts`-Metrik hängt von kalibrierten Wahrscheinlichkeiten ab. Der **Brier Score** misst die mittlere quadratische Abweichung zwischen vorhergesagter Wahrscheinlichkeit $p_i$ und tatsächlichem Ausgang $o_i$ (0 oder 1):

$B_{\text{Score}} = \frac{1}{N}\sum_{i=1}^{N} (p_i - o_i)^2$

Die Analyse in `xg_boost_modeling.ipynb` zeigte, dass XGBoost einen leicht besseren Brier Score aufwies, die Kalibrationskurven der Logistischen Regression jedoch monotoner und stabiler waren. Dies deutet darauf hin, dass XGBoost zu Overconfidence in dünn besetzten Datenregionen neigt.

**Spiel-aggregierte Validierung:**
Die entscheidende Validierung erfolgte auf Makro-Ebene durch Aggregation der `xPts` pro Spiel. Hier zeigte sich ein interessanter Trade-off:
*   **XGBoost** hatte eine höhere prädiktive Kraft auf Einzelwurf-Ebene (höherer ROC-AUC: 0.6686 vs. 0.6545).
*   Die **Logistische Regression** zeigte eine stärkere Korrelation ($R^2 = 0.205$ vs. $0.164$) zwischen der Summe der `xPts` und dem tatsächlichen Spielergebnis.

**Interpretation:** Das lineare Modell ist robuster gegenüber Ausreißern und seine Fehler mitteln sich auf Aggregatebene besser heraus. XGBoost erfasst zwar mehr Nuancen, seine Fehler können sich aber auch summieren. Für die *Analyse von Einzelentscheidungen* ist XGBoost überlegen, für die *Prognose von Spielergebnissen* die Logistische Regression. Da unser Ziel die Optimierung von *Entscheidungen* war, wurde XGBoost als Primärmodell gewählt.

### 2.3 Model & Daten: „Big Data adressiert“


*   **Skalierbarkeit:** Die saisonweise Datenakquise ist eine Form des **Batch Processing**, die eine Überlastung der API verhindert und die Verarbeitung großer Datenmengen ermöglicht. Das System ist konzeptionell für eine Migration auf eine verteilte Computing-Plattform wie Apache Spark vorbereitet, da die Logik bereits in transformator- und funktionsbasierten Modulen gekapselt ist.

*   **Automation:** Die Pipeline ist vollständig skriptbasiert. Die `utils`-Module entkoppeln die Implementierungsdetails von der Ausführungslogik in den `scripts`-Notebooks. Dies ermöglicht eine automatisierte Ausführung der gesamten Kette, von der Datenakquise über das Feature Engineering bis zum Modelltraining und der Evaluation.

*   **Minimierung des Memory Overheads:** Die Wahl von **Parquet** als Speicherformat reduziert den Speicherbedarf und die Ladezeiten erheblich. Die Verwendung von `pandas`-internen, optimierten Funktionen wie `rolling()` und `transform()` ist speichereffizienter als manuelle Implementierungen. Für ein noch größeres Datenvolumen könnten `pandas`-Iteratoren oder Frameworks wie `Polars` eingesetzt werden.


## Rollenverteilung:
Interdisziplinäres Team mit unterschiedlichen Schwerpunkten aber übergreifenden Kompetenzen. 

- **Enes**: EDA, Streamlit, Pipeline, Documentation (33.33%)
- **Furkan**: Pipeline, Feature Engineering, Research, Modeling (Logistic Regression), Documentation (33.33%)
- **Melih**: EDA, Research, Modeling (XGBoost), Documentation (33.33%)

Alle Teammitglieder haben gemeinsam über ihren Aufgabenbereich den selben Workload gehabt.