# CARE — Customer Analysis \& Response Engine

## Customer Feedback Analysis and Automated Response

**Imarticus Data Science Internship — Python Foundations \& Gen AI**

**Technology:** Python, Pandas, Rule-Based Text Analysis, `collections.Counter`, Regular Expressions, and Google Gemini

\---

## 1\. Project Overview

CARE (Customer Analysis \& Response Engine) is a Python-based customer feedback analysis pipeline developed for the Imarticus Data Science Internship assessment.

The notebook follows the three required stages of the assessment:

1. **Data Wrangling** — Load, inspect, validate, clean, and prepare the customer review dataset.
2. **Rule-Based Filtering \& Insights** — Identify Critical Reviews using a transparent rating rule and calculate common complaint keywords.
3. **Generative AI Automated Outreach** — Select three of the most critical and detailed negative reviews and generate personalized customer-support apology emails using Google Gemini.

The project intentionally keeps the analytical decision-making explainable. Machine Learning is not used for Critical Review detection or complaint-keyword analysis.

\---

# 2\. Dataset

The notebook uses the **Women's Clothing E-Commerce Customer Reviews Dataset**.

The source dataset contains:

* **23,486 records**
* **11 columns**

Important fields used by CARE include:

|Column|Purpose|
|-|-|
|`Clothing ID`|Product identifier|
|`Age`|Customer age|
|`Title`|Review title|
|`Review Text`|Original customer-written feedback|
|`Rating`|1–5 star customer rating|
|`Recommended IND`|Recommendation indicator|
|`Division Name`|Product division|
|`Department Name`|Product department|
|`Class Name`|Product class|

The dataset is loaded from a CSV file. The notebook allows the user to specify the local file path in the dataset-loading cell.

\---

# 3\. Data Cleaning and Preparation Approach

The cleaning process follows an evidence-based approach: **inspect → transform → recheck → confirm**.

The notebook does not make unnecessary assumptions about missing information.

## 3.1 Remove the exported index column

The original `Unnamed: 0` column represents an exported row index rather than a meaningful business attribute.

It is removed from the working dataset.

## 3.2 Remove duplicate records

Complete duplicate records are identified and removed using Pandas `drop\_duplicates()`.

The first occurrence of a duplicated record is retained.

The notebook records the number of rows before and after duplicate removal so that the transformation is traceable.

After duplicate removal, the working dataset contains:

**23,465 records**

## 3.3 Validate Rating

The `Rating` field is checked to ensure that values remain within the expected 1–5 range.

No invalid or missing Rating values require correction.

The original Rating values are therefore retained.

## 3.4 Validate Recommended IND

The `Recommended IND` field is checked against its expected binary values:

* `0`
* `1`

No invalid values require modification.

## 3.5 Handle missing Review Text

The notebook does **not** delete records simply because their review text is missing.

A separate indicator named:

`Has Review Text`

is created to distinguish records with usable customer-written feedback from records without it.

This preserves the complete review population because a review can still be Critical based on its rating even when no written text is available.

The cleaned dataset contains:

* **22,640 records with usable review text**
* **825 records with missing review text**

## 3.6 Preserve the original review text

Before text preprocessing, the original customer feedback is copied into:

`Original Review Text`

This provides traceability between the source review and its cleaned representation.

The original customer wording is never overwritten.

## 3.7 Handle missing Title values

Missing `Title` values are retained.

No artificial titles are created because a missing title does not prevent the core CARE workflow from operating.

## 3.8 Handle missing product categories

Missing values in:

* `Division Name`
* `Department Name`
* `Class Name`

are retained.

The notebook does not fabricate product categories because these fields are supporting variables and are not required for Critical Review classification, complaint-keyword analysis, or Generative AI outreach.

The affected records are therefore preserved as missing rather than being assigned invented values.

\---

# 4\. Text Preprocessing Approach

CARE creates a separate cleaned text field called:

`Clean Review Text`

The purpose is to standardize text for rule-based keyword analysis while preserving the customer's meaningful language.

The preprocessing performs the following:

### Lowercase conversion

Review text is converted to lowercase so that keyword matching is consistent.

For example:

`Poor FIT` becomes:

`poor fit`

### Whitespace normalization

Line breaks, tabs, and repeated whitespace are converted into a single space.

Leading and trailing whitespace is removed.

### Missing text handling

If `Review Text` is missing, the cleaning function returns an empty string.

### What is intentionally NOT done

CARE does not perform aggressive NLP transformations.

The notebook does **not**:

* Remove stopwords indiscriminately
* Stem words
* Aggressively lemmatize words
* Overwrite the original customer review
* Remove meaningful complaint language

This conservative approach is used because complaint expressions and negation words can carry important meaning.

The cleaned text is used for rule-based analysis, while `Original Review Text` remains available for traceability and Generative AI outreach.

\---

# 5\. Rule-Based Critical Review Logic

The assessment requires Critical Reviews to be identified without Machine Learning.

CARE uses the following deterministic rule:

> \*\*Critical Review = Rating 1 OR Rating 2\*\*

In Python:

```python
df\["Critical Review"] = df\["Rating"].isin(\[1, 2]).astype(int)
```

Therefore:

* Rating **1** → Critical
* Rating **2** → Critical
* Rating **3** → Not Critical
* Rating **4** → Not Critical
* Rating **5** → Not Critical

The notebook independently checks the calculated classification against the validated Rating field.

The result is:

* **2,407 Critical Reviews**
* **0 classification mismatches**
* **10.26% of the working dataset**

Among the Critical Reviews:

* **2,370 contain usable review text**
* **37 do not contain review text**

The 37 Critical Reviews without text are retained because Critical status is determined by rating, not by text availability.

\---

# 6\. Complaint Keyword Analysis

The second part of Task 2 requires identification of common complaint keywords using basic Python logic.

CARE uses a predefined complaint vocabulary containing terms related to:

* Fit and sizing
* Fabric and material
* Quality
* Appearance and color
* Product design
* Disappointment
* Physical product issues

Examples include:

`fit`, `size`, `small`, `large`, `fabric`, `material`, `quality`, `color`, `design`, `broken`, `pilling`, `shrink`, `seam`, and `stitching`.

Only Critical Reviews with usable review text are included in this analysis.

## Counting method

The notebook uses:

* Regular expressions
* Lowercase text
* Tokenization
* `collections.Counter`

The function:

```python
find\_common\_complaint\_keywords()
```

counts occurrences of the predefined complaint words and returns the most frequent 15 keywords.

No Machine Learning model is used.

## Calculated top complaint keywords

|Rank|Complaint Keyword|Frequency|
|-:|-|-:|
|1|`fabric`|712|
|2|`fit`|687|
|3|`size`|677|
|4|`small`|518|
|5|`material`|428|
|6|`color`|380|
|7|`quality`|339|
|8|`large`|302|
|9|`disappointed`|254|
|10|`cut`|253|
|11|`short`|242|
|12|`flattering`|236|
|13|`big`|229|
|14|`design`|216|
|15|`thin`|213|

These frequencies represent occurrences of the predefined keywords in the Critical Review text population. They should not be interpreted as the exact number of separate complaints or customers.

\---

# 7\. Selection of the Three Reviews for Generative AI

For Task 3, CARE selects exactly three of the most critical and detailed negative reviews.

The selection logic is transparent and reproducible:

1. Restrict the population to **1-star reviews**.
2. Keep only reviews with usable text.
3. Calculate **Word Count**.
4. Calculate **Character Count**.
5. Sort by:

   * Word Count descending
   * Character Count descending
   * Clothing ID ascending as a deterministic tie-breaker
6. Select the first three records using `head(3)`.

The use of 1-star reviews prioritizes the most severe cases within the assessment-defined 1–2 star Critical Review population.

Only these three reviews are passed to the Generative AI stage.

\---

# 8\. Generative AI Approach

Google Gemini is used for the final automated outreach stage.

The notebook uses the Google GenAI Python SDK and the model:

`gemini-3.1-flash-lite`

The API is called only after:

* Data cleaning is complete
* Critical Reviews have been identified
* Complaint keywords have been analyzed
* Three detailed 1-star reviews have been selected

## Customer Support Prompt

Gemini is instructed to act as a Customer Support Agent.

The prompt requires the generated response to:

* Acknowledge the customer's specific negative experience
* Address the complaint mentioned in the review
* Be polite and empathetic
* Remain short and clear
* Avoid arguing with or blaming the customer
* Avoid inventing refunds, replacements, discounts, policies, or other facts
* Avoid unsupported promises
* Return only the customer-facing apology email

Each selected review is sent individually so that the generated response is specific to that customer's feedback.

\---

# 9\. How to Run the Notebook

## Step 1 — Install the required packages

The notebook uses standard Python libraries together with Pandas and the Google GenAI SDK.

If the Gemini SDK is not already installed, run:

```bash
pip install -U google-genai
```

The main libraries used by the notebook are:

```text
pandas
google-genai
```

Python's built-in modules such as `re`, `time`, `collections`, and `pathlib` are also used.

\---

## Step 2 — Open the Jupyter Notebook

Open:

`CARE - Customer Analysis and Response Engine.ipynb`

in Jupyter Notebook or JupyterLab.

\---

## Step 3 — Enter the dataset path

In the dataset-loading cell, locate:

```python
file\_path = r"YOUR\_DATASET\_PATH"
```

Replace it with the actual location of:

`Womens Clothing E-Commerce Reviews.csv`

For example:

```python
file\_path = r"E:\\Internship Assessment\\archive\\Womens Clothing E-Commerce Reviews.csv"
```

Run the notebook from the beginning so that all required variables and cleaned datasets are created in the correct order.

\---

# 10\. Supplying the Gemini API Key

The Gemini API key is entered directly in the notebook's authentication cell in Section **16.5**.

The cell contains:

```python
GEMINI\_API\_KEY = "PASTE\_YOUR\_GEMINI\_API\_KEY\_HERE"
```

Replace the placeholder with your valid Gemini API key:

```python
GEMINI\_API\_KEY = "AQ.Ab8RN6L8OgrrffAtwTgy6Tpsrjk216qG69pJVA9QlctR03YvMw"
```

Then initialize the client:

```python
client = genai.Client(api\_key=GEMINI\_API\_KEY)
```

Run Section **16.5** before running the Generative AI generation cell.

### Important security note

Do **not** publish, upload, commit, or share a real API key in a public repository or submission file.

For the submitted README, only the placeholder/instructions should be included. The actual API key should remain private.

If an API key has accidentally been exposed publicly, it should be revoked and replaced with a new key.

\---

# 11\. Running the Generative AI Portion

After completing Sections 16.5 and 16.6:

1. Run **16.7 — Generate the 3 AI-Generated Response Emails**.
2. The notebook sends the three selected review texts individually to Gemini.
3. The notebook includes a short retry mechanism for temporary 503 server-availability errors.
4. Other API errors are allowed to surface normally.
5. The notebook verifies that all three responses were generated.

The expected final validation is:

```text
AI responses generated successfully: 3
```

If the API returns an authentication error such as `400 INVALID\_ARGUMENT` with `API\_KEY\_INVALID`, verify that the API key entered in Section 16.5 is valid.

\---

# 12\. Generated Outputs

After successful execution of Task 3, Section 16.8:

1. Displays the three actual Gemini-generated customer-support emails as rendered Markdown.
2. Shows the corresponding Clothing ID, Rating, Title, and original customer review.
3. Saves the generated responses to:

`AI\_Generated\_Responses.md`

The three responses are generated dynamically from the selected reviews and are not manually hard-coded into the notebook.

\---

# 13\. Recommended Execution Order

For the most reliable execution, run the notebook sequentially from top to bottom:

```text
Dataset Loading
       ↓
Data Inspection
       ↓
Data Cleaning
       ↓
Text Preprocessing
       ↓
Critical Review Detection
       ↓
Complaint Keyword Analysis
       ↓
Three Detailed Review Selection
       ↓
Gemini SDK Setup
       ↓
API Authentication
       ↓
Customer Support Prompt
       ↓
Generate 3 AI Responses
       ↓
Display and Save AI Responses
       ↓
Final Submission Checklist
```

Running cells out of order may cause variables such as `selected\_reviews\_df`, `generated\_responses`, or `client` to be unavailable.

\---

# 14\. Assessment Requirement Mapping

|Assessment Requirement|CARE Implementation|
|-|-|
|Source a customer review dataset|Women's Clothing E-Commerce Customer Reviews Dataset|
|Load using Pandas|`pd.read\_csv()`|
|Handle missing values|Missing text and supporting fields are assessed and handled without unnecessary deletion|
|Clean review text|Lowercase conversion and whitespace normalization|
|Identify Critical Reviews|1–2 star rule|
|No Machine Learning|Rule-based Python logic only for Task 2|
|Find common complaint keywords|Predefined complaint vocabulary + regex + `Counter`|
|Select 3 critical detailed reviews|1-star reviews ranked by word count and deterministic tie-breakers|
|Use a Generative AI API|Google Gemini|
|Generate personalized responses|Customer Support Agent prompt|
|Display calculated insights|Complaint keyword table and analytical findings|
|Display 3 AI responses|Section 16.8|
|Explain API-key setup|Section 16.5 and this README|

\---

# 15\. Final Submission Checklist

Before submitting the project, confirm:

* \[ ] The CSV dataset path is correct.
* \[ ] The notebook runs successfully from the beginning.
* \[ ] The data-cleaning steps execute without errors.
* \[ ] The Critical Review count is calculated correctly.
* \[ ] The complaint keyword table is generated.
* \[ ] Exactly three detailed 1-star reviews are selected.
* \[ ] A valid Gemini API key is supplied privately.
* \[ ] Section 16.7 successfully generates exactly three responses.
* \[ ] Section 16.8 displays the three generated emails.
* \[ ] `AI\_Generated\_Responses.md` is created after successful API execution.
* \[ ] No real API key is included in the README or public submission.

\---

## 16\. Project Deliverables

The project consists of:

1. **Jupyter Notebook**  
`CARE - Customer Analysis and Response Engine.ipynb`
2. **README**  
This document.
3. **AI Response Artifact**  
`AI\_Generated\_Responses.md` — automatically created after successful execution of the Gemini generation stage.

\---

## 17\. Summary

CARE demonstrates an end-to-end customer feedback workflow using Python and Generative AI.

The analytical process begins with structured data cleaning and validation, followed by a transparent rule that identifies 1–2 star reviews as Critical. The Critical Review population is then analyzed using predefined complaint keywords and `collections.Counter` to identify recurring complaint signals.

For the final outreach stage, the three most critical and detailed 1-star reviews with usable text are selected using a reproducible ranking method. Google Gemini then generates short, personalized, and empathetic customer-support apology emails based on the actual review content.

The overall workflow is intentionally explainable, reproducible, and aligned with the three tasks specified in the Imarticus Data Science Internship assessment.

