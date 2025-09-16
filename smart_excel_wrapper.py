import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class SmartExcelWrapper:
    def __init__(self, model_path="./qwen_excel_gpu"):
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
        
        # Add the full name
        keywords.append(col_lower)
        
        # Add common aliases
        aliases = {
            'sales': ['revenue', 'income', 'earnings'],
            'year': ['date', 'time', 'period'],
            'name': ['title', 'label'],
            'publisher': ['company', 'developer', 'studio'],
            'global': ['total', 'worldwide', 'overall'],
            'na': ['north america', 'usa', 'us'],
            'eu': ['europe', 'european']
        }
        
        for part in parts:
            if part in aliases:
                keywords.extend(aliases[part])
                
        return list(set(keywords))
    
    def find_target_column(self, query):
        """Enhanced column detection from query"""
        query_lower = query.lower().replace('_', ' ').replace('-', ' ')
        best_match = None
        best_score = 0
        target_value = None
        
        # Look for specific values first
        for col_key, info in self.column_info.items():
            for value in info['unique_values']:
                if isinstance(value, str) and value.lower() in query_lower:
                    return info['letter'], info['name'], value
        
        # Score-based matching for column names
        for col_key, info in self.column_info.items():
            score = 0
            
            # Direct name match
            if info['name'].lower() in query_lower:
                score += 10
                
            # Keyword matches
            for keyword in info['keywords']:
                if keyword in query_lower:
                    score += len(keyword)  # Longer matches get higher scores
            
            # Pattern-based matches
            if any(word in query_lower for word in ['percentage', 'percent']) and 'sales' in info['keywords']:
                score += 5
            
            if score > best_score:
                best_score = score
                best_match = info
        
        if best_match:
            return best_match['letter'], best_match['name'], target_value
        
        return None, None, None
    
    def generate_smart_formula(self, query):
        """Generate formula - prioritize fine-tuned model for clear Excel requests"""
        if self.df is None:
            return "Please upload a CSV file first."
        
        # Use fine-tuned model first for Excel-specific requests
        if self.model_loaded and self._is_clear_excel_request(query):
            formula = self._use_trained_model(query)
            if formula and formula.startswith('=') and 'error' not in formula.lower():
                return formula
        
        # Fallback to rule-based for specific patterns
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['percentage', 'percent', 'rate', '%']):
            return self._generate_percentage_formula(query)
        elif any(word in query_lower for word in ['count', 'how many', 'number of', 'total number']):
            return self._generate_count_formula(query)
        elif any(word in query_lower for word in ['sum', 'total', 'add up', 'combined']):
            return self._generate_sum_formula(query)
        elif any(word in query_lower for word in ['average', 'mean', 'avg']):
            return self._generate_average_formula(query)
        elif any(word in query_lower for word in ['max', 'maximum', 'highest', 'largest']):
            return self._generate_max_formula(query)
        elif any(word in query_lower for word in ['min', 'minimum', 'lowest', 'smallest']):
            return self._generate_min_formula(query)
        else:
            return "Could not generate formula for this query."
    
    def _is_clear_excel_request(self, query):
        """Check if query is clearly asking for Excel formula"""
        excel_indicators = [
            'formula', 'calculate', '=', 'sum of', 'count of', 'average of',
            'percentage of', 'total of', 'max of', 'min of', 'excel'
        ]
        return any(indicator in query.lower() for indicator in excel_indicators)
    
    def _generate_percentage_formula(self, query):
        """Enhanced percentage formulas"""
        col_letter, col_name, target_value = self.find_target_column(query)
        
        if col_letter and target_value:
            return f'=COUNTIF({col_letter}:{col_letter},"{target_value}")/COUNTA({col_letter}:{col_letter})*100'
        elif col_letter:
            # Try to find a specific value mentioned in the query
            query_words = query.lower().split()
            for word in query_words:
                for val in self.column_info[col_name.lower().replace(' ', '_')]['unique_values']:
                    if isinstance(val, str) and word.lower() in val.lower():
                        return f'=COUNTIF({col_letter}:{col_letter},"*{word}*")/COUNTA({col_letter}:{col_letter})*100'
            
            # Default to first unique value
            col_info_key = next((k for k in self.column_info.keys() if self.column_info[k]['name'] == col_name), None)
            if col_info_key and self.column_info[col_info_key]['unique_values']:
                first_value = self.column_info[col_info_key]['unique_values'][0]
                return f'=COUNTIF({col_letter}:{col_letter},"{first_value}")/COUNTA({col_letter}:{col_letter})*100'
        
        return "Could not generate percentage formula."
    
    def _generate_count_formula(self, query):
        """Enhanced count formulas"""
        col_letter, col_name, target_value = self.find_target_column(query)
        
        if col_letter and target_value:
            return f'=COUNTIF({col_letter}:{col_letter},"{target_value}")'
        elif col_letter:
            # Look for specific values or patterns in query
            query_words = query.lower().split()
            for word in query_words:
                # Check if word is a year
                if word.isdigit() and len(word) == 4:
                    return f'=COUNTIF({col_letter}:{col_letter},{word})'
                # Check if word matches any unique values
                col_info_key = next((k for k in self.column_info.keys() if self.column_info[k]['name'] == col_name), None)
                if col_info_key:
                    for val in self.column_info[col_info_key]['unique_values']:
                        if isinstance(val, str) and word.lower() in val.lower():
                            return f'=COUNTIF({col_letter}:{col_letter},"*{word}*")'
            
            return f'=COUNTA({col_letter}:{col_letter})'
        
        return "Could not generate count formula."
    
    def _generate_sum_formula(self, query):
        """Enhanced sum formulas"""
        col_letter, col_name, _ = self.find_target_column(query)
        
        if col_letter:
            col_info_key = next((k for k in self.column_info.keys() if self.column_info[k]['name'] == col_name), None)
            if col_info_key:
                col_info = self.column_info[col_info_key]
                if 'int' in col_info['type'] or 'float' in col_info['type']:
                    return f'=SUM({col_letter}:{col_letter})'
        
        # Find any numeric column if no specific match
        for col_key, info in self.column_info.items():
            if 'int' in info['type'] or 'float' in info['type']:
                return f'=SUM({info["letter"]}:{info["letter"]})'
        
        return "Could not generate sum formula."
    
    def _generate_average_formula(self, query):
        """Enhanced average formulas"""
        col_letter, col_name, _ = self.find_target_column(query)
        
        if col_letter:
            col_info_key = next((k for k in self.column_info.keys() if self.column_info[k]['name'] == col_name), None)
            if col_info_key:
                col_info = self.column_info[col_info_key]
                if 'int' in col_info['type'] or 'float' in col_info['type']:
                    return f'=AVERAGE({col_letter}:{col_letter})'
        
        for col_key, info in self.column_info.items():
            if 'int' in info['type'] or 'float' in info['type']:
                return f'=AVERAGE({info["letter"]}:{info["letter"]})'
        
        return "Could not generate average formula."
    
    def _generate_max_formula(self, query):
        """Enhanced max formulas"""
        col_letter, col_name, _ = self.find_target_column(query)
        
        if col_letter:
            col_info_key = next((k for k in self.column_info.keys() if self.column_info[k]['name'] == col_name), None)
            if col_info_key:
                col_info = self.column_info[col_info_key]
                if 'int' in col_info['type'] or 'float' in col_info['type']:
                    return f'=MAX({col_letter}:{col_letter})'
        
        for col_key, info in self.column_info.items():
            if 'int' in info['type'] or 'float' in info['type']:
                return f'=MAX({info["letter"]}:{info["letter"]})'
        
        return "Could not generate max formula."
    
    def _generate_min_formula(self, query):
        """Enhanced min formulas"""
        col_letter, col_name, _ = self.find_target_column(query)
        
        if col_letter:
            col_info_key = next((k for k in self.column_info.keys() if self.column_info[k]['name'] == col_name), None)
            if col_info_key:
                col_info = self.column_info[col_info_key]
                if 'int' in col_info['type'] or 'float' in col_info['type']:
                    return f'=MIN({col_letter}:{col_letter})'
        
        for col_key, info in self.column_info.items():
            if 'int' in info['type'] or 'float' in info['type']:
                return f'=MIN({info["letter"]}:{info["letter"]})'
        
        return "Could not generate min formula."
    
    def _use_trained_model(self, query):
        """Use trained model with focused prompts for Excel formulas only"""
        if not self.model_loaded:
            return None
        
        # Only use for clear Excel formula requests
        if not self._is_clear_excel_request(query):
            return None
        
        if self.df is not None:
            # Focused prompt for formula generation
            columns = list(self.df.columns)
            column_mapping = {col: chr(65 + i) for i, col in enumerate(columns)}
            
            enhanced_prompt = f"""<|im_start|>system
You are an Excel formula expert. Generate only Excel formulas.
<|im_end|>
<|im_start|>user
Dataset columns: {columns}
Column letters: {column_mapping}
Generate Excel formula for: {query}
<|im_end|>
<|im_start|>assistant
="""
        else:
            enhanced_prompt = f"""<|im_start|>system
You are an Excel formula expert.
<|im_end|>
<|im_start|>user
Generate Excel formula for: {query}
<|im_end|>
<|im_start|>assistant
="""
        
        try:
            inputs = self.tokenizer(enhanced_prompt, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=40,  # Short output for formulas
                    temperature=0.05,   
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            new_tokens = outputs[0][len(inputs['input_ids'][0]):]
            formula = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Clean up formula
            formula = formula.replace("<|im_end|>", "").strip()
            formula = re.sub(r'<\|.*?\|>', '', formula).strip()
            
            # Take only the first line
            formula = formula.split('\n')[0].strip()
            
            # Ensure it starts with =
            if formula and not formula.startswith('='):
                formula = f'={formula}'
            
            return formula if formula and len(formula) > 1 else None
            
        except Exception as e:
            print(f"Error using trained model: {e}")
            return None
    
    def get_column_info(self):
        """Get detailed column information"""
        if self.df is None:
            return "No CSV loaded."
        
        info = f"Dataset: {len(self.df)} rows, {len(self.df.columns)} columns\n\n"
        info += "Available Columns:\n\n"
        
        for col_key, details in self.column_info.items():
            info += f"• **{details['name']}** (Column {details['letter']})\n"
            info += f"  Type: {details['type']}\n"
            info += f"  Keywords: {', '.join(details['keywords'][:5])}\n"
            if details['unique_values']:
                values = ', '.join([str(v)[:15] for v in details['unique_values'][:3]])
                info += f"  Sample values: {values}\n"
            info += "\n"
        
        return info