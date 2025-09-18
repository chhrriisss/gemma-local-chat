import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class SmartExcelWrapper:
    def __init__(self, model_path="./qwen_excel_reasoning_gpu"):
        """Load your trained model with smart column mapping"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.model.eval()
            self.model_loaded = True
            print(f"✅ Fine-tuned model loaded from {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load fine-tuned model: {e}")
            self.model_loaded = False
        
        # Store current CSV info
        self.df = None
        self.column_mapping = {}
        self.column_info = {}
        
    def update_csv(self, df):
        """Update with new CSV structure"""
        self.df = df
        self.column_mapping = {col: chr(65 + i) for i, col in enumerate(df.columns)}
        
        # Analyze columns with more detail
        self.column_info = {}
        for col in df.columns:
            col_lower = col.lower().replace('_', ' ').replace('-', ' ')
            self.column_info[col_lower] = {
                'letter': self.column_mapping[col],
                'name': col,
                'type': str(df[col].dtype),
                'unique_values': df[col].dropna().unique()[:10].tolist(),
                'keywords': self._extract_keywords(col)
            }
    
    def _extract_keywords(self, column_name):
        """Extract keywords from column name for better matching"""
        keywords = []
        col_lower = column_name.lower()
        
        # Split by common separators
        parts = re.split(r'[_\-\s]+', col_lower)
        keywords.extend(parts)
        keywords.append(col_lower)
        
        return list(set(keywords))
    
    def generate_comprehensive_answer(self, query):
        """Main method that handles ALL CSV queries with consistency"""
        if self.df is None:
            return "Please upload a CSV file first."
        
        # First, try the trained model
        if self.model_loaded:
            response = self._use_trained_model(query)
            
            # Apply consistency fix to the response
            if response:
                fixed_response = self._ensure_consistent_response(query, response)
                return fixed_response
        
        # Fallback to rule-based if model fails
        return self._generate_fallback_response(query)
    
    def _ensure_consistent_response(self, query, model_response):
        """Universal consistency fix for any dataset"""
        query_lower = query.lower()
        response_lower = model_response.lower()
        
        # Check if model gave a "cannot answer" type response
        cannot_answer_phrases = [
            "cannot answer", "don't have", "not specify", "unable to", 
            "not available", "cannot determine", "insufficient data",
            "does not specify", "not provided", "cannot provide"
        ]
        
        if any(phrase in response_lower for phrase in cannot_answer_phrases):
            
            # Check what type of query it is and provide appropriate formula
            if self._is_counting_query(query):
                return self._generate_counting_formula_response(query)
            elif self._is_percentage_query(query):
                return self._generate_percentage_formula_response(query)
            elif self._is_summary_query(query):
                return self._generate_summary_response(query)
            elif self._is_statistical_query(query):
                return self._generate_statistical_response(query)
        
        # If response looks good, return as is
        return model_response
    
    def _is_counting_query(self, query):
        """Check if query is asking to count something"""
        counting_indicators = [
            "how many", "count", "number of", "times does", "appears", 
            "occur", "instances", "frequency", "total number"
        ]
        return any(indicator in query.lower() for indicator in counting_indicators)
    
    def _is_percentage_query(self, query):
        """Check if query is asking for percentage"""
        return any(word in query.lower() for word in ["percentage", "percent", "%", "proportion"])
    
    def _is_summary_query(self, query):
        """Check if query is asking for dataset summary"""
        summary_indicators = [
            "summary", "summarize", "overview", "describe", "structure",
            "columns", "what data", "what's in", "tell me about"
        ]
        return any(indicator in query.lower() for indicator in summary_indicators)
    
    def _is_statistical_query(self, query):
        """Check if query is asking for statistics"""
        stats_indicators = [
            "average", "mean", "total", "sum", "maximum", "minimum", 
            "highest", "lowest", "statistics", "stats"
        ]
        return any(indicator in query.lower() for indicator in stats_indicators)
    
    def _generate_counting_formula_response(self, query):
        """Generate counting formula for any value mentioned in query"""
        # Try to find what they want to count
        target_column, target_value = self._extract_count_target(query)
        
        if target_column and target_value:
            return f"**Excel Formula:** `=COUNTIF({target_column}:{target_column},\"{target_value}\")`\n\n**Explanation:** This formula counts all occurrences of \"{target_value}\" in the {self._get_column_name(target_column)} column.\n\n**Reasoning:** COUNTIF searches through the specified column and counts cells that match the exact criteria."
        
        elif target_column:
            return f"**Excel Formula:** `=COUNTA({target_column}:{target_column})`\n\n**Explanation:** This formula counts all non-empty entries in the {self._get_column_name(target_column)} column.\n\n**Reasoning:** COUNTA counts all cells containing data, giving the total number of entries."
        
        else:
            # Generic counting response
            return f"**Excel Formula:** `=COUNTA(A:A)`\n\n**Explanation:** This formula counts entries in the first column. Adjust the column reference (A:A) to match your specific data.\n\n**Reasoning:** Use COUNTIF for specific values or COUNTA for total entries in any column."
    
    def _generate_percentage_formula_response(self, query):
        """Generate percentage formula for any value mentioned in query"""
        target_column, target_value = self._extract_count_target(query)
        
        if target_column and target_value:
            return f"**Excel Formula:** `=COUNTIF({target_column}:{target_column},\"{target_value}\")/COUNTA({target_column}:{target_column})*100`\n\n**Explanation:** This calculates what percentage of records have \"{target_value}\" in the {self._get_column_name(target_column)} column.\n\n**Reasoning:** Divides the count of matching entries by total entries and multiplies by 100 to get percentage."
        
        else:
            return "**Excel Formula:** `=COUNTIF(Column:Column,\"Value\")/COUNTA(Column:Column)*100`\n\n**Explanation:** Replace 'Column:Column' with your target column and 'Value' with your target value.\n\n**Reasoning:** This formula calculates percentage by dividing specific count by total count."
    
    def _generate_summary_response(self, query):
        """Generate dataset summary"""
        if self.df is not None:
            summary = f"**Dataset Summary:**\n\n"
            summary += f"• **Structure:** {len(self.df)} rows × {len(self.df.columns)} columns\n"
            summary += f"• **Columns:** {', '.join(self.df.columns)}\n\n"
            
            # Add column details
            summary += "**Column Details:**\n"
            for col_key, info in self.column_info.items():
                summary += f"• **{info['name']}** (Column {info['letter']}): {info['type']}"
                if info['unique_values']:
                    values = ', '.join([str(v) for v in info['unique_values'][:3]])
                    summary += f" - Sample values: {values}"
                summary += "\n"
            
            summary += f"\n**Reasoning:** This summary analyzes the dataset structure, identifying all columns and their data types for comprehensive understanding."
            
            return summary
        
        return "Please upload a CSV file to generate a summary."
    
    def _generate_statistical_response(self, query):
        """Generate statistical analysis response"""
        query_lower = query.lower()
        
        # Find numeric column for statistics
        numeric_column = self._find_best_numeric_column(query)
        
        if numeric_column:
            col_letter, col_name = numeric_column
            
            if "average" in query_lower or "mean" in query_lower:
                return f"**Excel Formula:** `=AVERAGE({col_letter}:{col_letter})`\n\n**Explanation:** Calculates the average of all values in the {col_name} column.\n\n**Reasoning:** AVERAGE function computes the arithmetic mean by summing all values and dividing by count."
            
            elif "sum" in query_lower or "total" in query_lower:
                return f"**Excel Formula:** `=SUM({col_letter}:{col_letter})`\n\n**Explanation:** Calculates the total sum of all values in the {col_name} column.\n\n**Reasoning:** SUM function adds all numerical values to provide the cumulative total."
            
            elif "max" in query_lower or "highest" in query_lower:
                return f"**Excel Formula:** `=MAX({col_letter}:{col_letter})`\n\n**Explanation:** Finds the highest value in the {col_name} column.\n\n**Reasoning:** MAX function identifies the largest numerical value in the specified range."
            
            elif "min" in query_lower or "lowest" in query_lower:
                return f"**Excel Formula:** `=MIN({col_letter}:{col_letter})`\n\n**Explanation:** Finds the lowest value in the {col_name} column.\n\n**Reasoning:** MIN function identifies the smallest numerical value in the specified range."
        
        return "**Analysis:** Statistical calculations require numeric data columns. Please specify which column you'd like to analyze.\n\n**Reasoning:** Different statistical measures (average, sum, max, min) can be applied to numerical data using appropriate Excel functions."
    
    def _extract_count_target(self, query):
        """Extract what to count from the query"""
        query_words = query.lower().split()
        
        # Look for quoted values or specific terms
        for col_key, info in self.column_info.items():
            # Check if column name is mentioned
            if info['name'].lower() in query.lower():
                # Look for specific values in that column
                for value in info['unique_values']:
                    if isinstance(value, str) and any(word in value.lower().split() for word in query_words):
                        return info['letter'], value
                
                # Return column without specific value
                return info['letter'], None
        
        # Try to extract any quoted or capitalized terms as potential values
        quoted_match = re.search(r'["\']([^"\']+)["\']', query)
        if quoted_match:
            potential_value = quoted_match.group(1)
            # Find most likely column
            best_col = self._find_most_likely_column(query)
            if best_col:
                return best_col[0], potential_value
        
        return None, None
    
    def _find_most_likely_column(self, query):
        """Find the most likely column based on query context"""
        query_lower = query.lower()
        
        # Simple scoring based on column name relevance
        best_score = 0
        best_col = None
        
        for col_key, info in self.column_info.items():
            score = 0
            if info['name'].lower() in query_lower:
                score += 10
            
            for keyword in info['keywords']:
                if keyword in query_lower:
                    score += len(keyword)
            
            if score > best_score:
                best_score = score
                best_col = (info['letter'], info['name'])
        
        return best_col
    
    def _find_best_numeric_column(self, query):
        """Find the best numeric column for statistical operations"""
        # Look for numeric columns
        for col_key, info in self.column_info.items():
            if 'int' in info['type'] or 'float' in info['type']:
                # Prefer columns mentioned in query
                if info['name'].lower() in query.lower():
                    return info['letter'], info['name']
        
        # Return first numeric column if none mentioned specifically
        for col_key, info in self.column_info.items():
            if 'int' in info['type'] or 'float' in info['type']:
                return info['letter'], info['name']
        
        return None
    
    def _get_column_name(self, column_letter):
        """Get column name from letter"""
        for col_key, info in self.column_info.items():
            if info['letter'] == column_letter:
                return info['name']
        return f"Column {column_letter}"
    
    def _use_trained_model(self, query):
        """Use the trained model with improved prompting"""
        if not self.model_loaded:
            return None
        
        # Enhanced prompt that encourages consistent formula responses
        if self.df is not None:
            columns = list(self.df.columns)
            column_mapping = {col: chr(65 + i) for i, col in enumerate(columns)}
            
            enhanced_prompt = f"""<|im_start|>system
You are an expert CSV data analyst. Always provide Excel formulas and explanations for data queries. Never say you cannot answer - instead provide the appropriate formula to analyze the data.
<|im_end|>
<|im_start|>user
Dataset columns: {columns}
Column letters: {column_mapping}

Query: {query}
<|im_end|>
<|im_start|>assistant
"""
        else:
            enhanced_prompt = f"""<|im_start|>system
You are an expert CSV data analyst. Always provide Excel formulas and explanations for data queries.
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
        
        try:
            inputs = self.tokenizer(enhanced_prompt, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # Increased for full responses
                    temperature=0.1,   
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            new_tokens = outputs[0][len(inputs['input_ids'][0]):]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Clean up response
            response = response.replace("<|im_end|>", "").strip()
            response = re.sub(r'<\|.*?\|>', '', response).strip()
            
            return response if response else None
            
        except Exception as e:
            print(f"Error using trained model: {e}")
            return None
    
    def _generate_fallback_response(self, query):
        """Fallback response generation"""
        return "I can help you analyze this CSV data. Please specify what you'd like to calculate, count, or analyze from the dataset."
    
    def get_column_info(self):
        """Get detailed column information"""
        if self.df is None:
            return "No CSV loaded."
        
        info = f"Dataset: {len(self.df)} rows, {len(self.df.columns)} columns\n\n"
        info += "Available Columns:\n\n"
        
        for col_key, details in self.column_info.items():
            info += f"• **{details['name']}** (Column {details['letter']})\n"
            info += f"  Type: {details['type']}\n"
            if details['unique_values']:
                values = ', '.join([str(v)[:15] for v in details['unique_values'][:3]])
                info += f"  Sample values: {values}\n"
            info += "\n"
        
        return info