export interface Statement {
  full_statement: string;
  class: string;
  deontic: string;
  action: string;
  conditions: string;
}

export interface Context {
  original_text: string;
}

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
}