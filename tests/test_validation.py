#!/usr/bin/env python3
"""
Unit tests for molecular utilities and validation
Tests RDKit integration, SMILES/SELFIES conversion, and filtering
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Test imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import selfies as sf
    SELFIES_AVAILABLE = True
except ImportError:
    SELFIES_AVAILABLE = False

# Sample test data
VALID_SMILES = [
    "CCO",  # ethanol
    "CC(=O)O",  # acetic acid
    "C1=CC=CC=C1",  # benzene
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # ibuprofen
]

INVALID_SMILES = [
    "CCC(",  # unclosed parenthesis
    "C1CC",  # unclosed ring
    "C[Zzz]",  # invalid atom
    "",  # empty string
    "X1XXXXX1"  # invalid pattern
]

class TestMolecularValidation:
    """Test molecular validation and sanitization."""
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_valid_smiles_parsing(self):
        """Test that valid SMILES are parsed correctly."""
        for smiles in VALID_SMILES:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Failed to parse valid SMILES: {smiles}"
            
            # Test sanitization
            try:
                Chem.SanitizeMol(mol)
                canonical_smiles = Chem.MolToSmiles(mol)
                assert canonical_smiles, f"Failed to canonicalize: {smiles}"
            except Exception as e:
                pytest.fail(f"Sanitization failed for {smiles}: {e}")
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_invalid_smiles_handling(self):
        """Test that invalid SMILES are handled gracefully."""
        for smiles in INVALID_SMILES:
            mol = Chem.MolFromSmiles(smiles)
            # Invalid SMILES should return None or raise exception during sanitization
            if mol is not None:
                # Some invalid SMILES might parse but fail sanitization
                with pytest.raises(Exception):
                    Chem.SanitizeMol(mol)
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_molecular_properties(self):
        """Test calculation of molecular properties."""
        test_cases = [
            ("CCO", {"mw_range": (40, 50), "logp_range": (-1, 0)}),
            ("C1=CC=CC=C1", {"mw_range": (75, 85), "logp_range": (1.5, 2.5)}),
            ("CC(=O)O", {"mw_range": (55, 65), "logp_range": (-1, 0)})
        ]
        
        for smiles, expected in test_cases:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            
            # Molecular weight
            mw = Descriptors.MolWt(mol)
            assert expected["mw_range"][0] <= mw <= expected["mw_range"][1], \
                f"MW {mw} not in expected range for {smiles}"
            
            # LogP
            logp = Descriptors.MolLogP(mol)
            assert expected["logp_range"][0] <= logp <= expected["logp_range"][1], \
                f"LogP {logp} not in expected range for {smiles}"

class TestSELFIESIntegration:
    """Test SELFIES integration and conversion."""
    
    @pytest.mark.skipif(not SELFIES_AVAILABLE, reason="SELFIES not available")
    def test_smiles_to_selfies_conversion(self):
        """Test conversion from SMILES to SELFIES."""
        for smiles in VALID_SMILES:
            try:
                selfies = sf.encoder(smiles)
                assert selfies, f"Failed to convert SMILES to SELFIES: {smiles}"
                assert selfies.startswith('[') and selfies.endswith(']'), \
                    f"Invalid SELFIES format: {selfies}"
            except Exception as e:
                pytest.fail(f"SELFIES encoding failed for {smiles}: {e}")
    
    @pytest.mark.skipif(not SELFIES_AVAILABLE, reason="SELFIES not available")
    def test_selfies_to_smiles_roundtrip(self):
        """Test round-trip conversion SMILES -> SELFIES -> SMILES."""
        for original_smiles in VALID_SMILES:
            try:
                # Convert to SELFIES
                selfies = sf.encoder(original_smiles)
                
                # Convert back to SMILES
                reconstructed_smiles = sf.decoder(selfies)
                
                # Verify both are valid and equivalent
                if RDKIT_AVAILABLE:
                    original_mol = Chem.MolFromSmiles(original_smiles)
                    reconstructed_mol = Chem.MolFromSmiles(reconstructed_smiles)
                    
                    assert original_mol is not None
                    assert reconstructed_mol is not None
                    
                    # Check if molecules are equivalent (same canonical SMILES)
                    original_canonical = Chem.MolToSmiles(original_mol)
                    reconstructed_canonical = Chem.MolToSmiles(reconstructed_mol)
                    
                    assert original_canonical == reconstructed_canonical, \
                        f"Round-trip failed: {original_smiles} -> {selfies} -> {reconstructed_smiles}"
                
            except Exception as e:
                pytest.fail(f"Round-trip conversion failed for {original_smiles}: {e}")

class TestMolecularFilters:
    """Test molecular filtering and drug-likeness rules."""
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_lipinski_rule_of_five(self):
        """Test Lipinski's Rule of Five implementation."""
        # Test molecules that should pass
        drug_like = ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O"]  # ethanol, aspirin
        
        # Test molecules that might fail (large, complex)
        non_drug_like = ["C" * 50]  # very long chain
        
        for smiles in drug_like:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            # Lipinski's Rule of Five (at least most should pass)
            lipinski_violations = 0
            if mw > 500:
                lipinski_violations += 1
            if logp > 5:
                lipinski_violations += 1
            if hbd > 5:
                lipinski_violations += 1
            if hba > 10:
                lipinski_violations += 1
            
            # Allow up to 1 violation for drug-like molecules
            assert lipinski_violations <= 1, \
                f"Too many Lipinski violations for {smiles}: MW={mw}, LogP={logp}, HBD={hbd}, HBA={hba}"
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_pains_filter_simulation(self):
        """Test PAINS (Pan Assay Interference Compounds) filter simulation."""
        # Common PAINS patterns (simplified)
        suspicious_patterns = [
            "C(=O)C(=O)",  # dicarbonyl
            "C=CC=CC=C",  # extended conjugation
        ]
        
        # Test molecules
        test_molecules = [
            "CCO",  # should be clean
            "CC(=O)C(=O)CC",  # contains dicarbonyl (potential PAINS)
        ]
        
        for smiles in test_molecules:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            
            # Check for suspicious patterns
            pains_count = 0
            for pattern in suspicious_patterns:
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                    pains_count += 1
            
            # First molecule should be clean, second might have issues
            if smiles == "CCO":
                assert pains_count == 0, f"Clean molecule flagged as PAINS: {smiles}"

class TestTokenizerIntegration:
    """Test tokenizer integration and molecular string handling."""
    
    def test_smiles_tokenization_mock(self):
        """Test SMILES tokenization with mock tokenizer."""
        # Mock tokenizer for testing
        class MockTokenizer:
            def __init__(self):
                self.vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "C": 3, "=": 4, "O": 5}
                self.pad_token_id = 0
                self.bos_token_id = 1
                self.eos_token_id = 2
            
            def encode(self, text, add_special_tokens=True, **kwargs):
                # Simple character-level tokenization for testing
                tokens = [self.bos_token_id] if add_special_tokens else []
                
                for char in text:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.vocab.get("[UNK]", len(self.vocab)))
                
                if add_special_tokens:
                    tokens.append(self.eos_token_id)
                
                return tokens
            
            def decode(self, token_ids, skip_special_tokens=True):
                reverse_vocab = {v: k for k, v in self.vocab.items()}
                chars = []
                
                for token_id in token_ids:
                    if token_id in reverse_vocab:
                        char = reverse_vocab[token_id]
                        if skip_special_tokens and char.startswith('['):
                            continue
                        chars.append(char)
                
                return ''.join(chars)
        
        tokenizer = MockTokenizer()
        
        # Test encoding
        test_smiles = "CCO"
        encoded = tokenizer.encode(test_smiles)
        
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        assert encoded[0] == tokenizer.bos_token_id
        assert encoded[-1] == tokenizer.eos_token_id
        
        # Test decoding
        decoded = tokenizer.decode(encoded)
        # Should get back original SMILES (character-level)
        assert decoded == test_smiles

class TestMemoryOptimization:
    """Test memory optimization and GPU compatibility."""
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_batch_processing_memory_limit(self):
        """Test batch processing with memory constraints."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process molecules in small batches (GTX 1650 simulation)
        molecules = VALID_SMILES * 20  # 100 molecules
        batch_size = 4
        
        processed_count = 0
        max_memory_increase = 0
        
        for i in range(0, len(molecules), batch_size):
            batch = molecules[i:i + batch_size]
            
            # Process batch
            batch_mols = []
            for smiles in batch:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    batch_mols.append(mol)
            
            processed_count += len(batch_mols)
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            max_memory_increase = max(max_memory_increase, memory_increase)
            
            # Clean up
            del batch_mols
            gc.collect()
        
        # Verify processing completed
        assert processed_count > 0, "No molecules were processed"
        
        # Memory should not increase dramatically (< 500MB for this test)
        assert max_memory_increase < 500, \
            f"Memory usage increased too much: {max_memory_increase:.1f} MB"
    
    def test_gpu_memory_simulation(self):
        """Test GPU memory management simulation."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        # Simulate GTX 1650 4GB VRAM constraint
        max_tensor_size_gb = 1.0  # Conservative limit
        max_elements = int(max_tensor_size_gb * 1024 * 1024 * 1024 / 4)  # float32
        
        # Test that we can create tensors within memory limit
        try:
            test_tensor = torch.randn(max_elements // 10)  # 10% of limit
            assert test_tensor.numel() > 0
            del test_tensor
            
            # Test that very large tensors would be problematic
            with pytest.raises((RuntimeError, MemoryError, torch.cuda.OutOfMemoryError)):
                huge_tensor = torch.randn(max_elements * 2)  # 2x the limit
                
        except Exception as e:
            # On CPU-only systems, memory errors might not occur the same way
            if "CUDA" not in str(e):
                pytest.skip(f"GPU memory test not applicable: {e}")

class TestValidationPipeline:
    """Test complete molecular validation pipeline."""
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_end_to_end_validation(self):
        """Test complete validation pipeline."""
        # Input molecules with mix of valid and invalid
        input_molecules = VALID_SMILES + INVALID_SMILES
        
        # Validation pipeline
        results = {
            'input_count': len(input_molecules),
            'valid_molecules': [],
            'invalid_molecules': [],
            'sanitization_errors': [],
            'property_calculations': []
        }
        
        for smiles in input_molecules:
            try:
                # Parse molecule
                mol = Chem.MolFromSmiles(smiles)
                
                if mol is None:
                    results['invalid_molecules'].append((smiles, "Parsing failed"))
                    continue
                
                # Sanitize
                try:
                    Chem.SanitizeMol(mol)
                except Exception as e:
                    results['sanitization_errors'].append((smiles, str(e)))
                    continue
                
                # Calculate properties
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                
                # Validate properties
                if 50 <= mw <= 1000 and -5 <= logp <= 10:
                    canonical_smiles = Chem.MolToSmiles(mol)
                    results['valid_molecules'].append(canonical_smiles)
                    results['property_calculations'].append({
                        'smiles': canonical_smiles,
                        'mw': mw,
                        'logp': logp
                    })
                else:
                    results['invalid_molecules'].append((smiles, "Properties out of range"))
                    
            except Exception as e:
                results['invalid_molecules'].append((smiles, f"Unexpected error: {e}"))
        
        # Validate results
        assert len(results['valid_molecules']) > 0, "No valid molecules found"
        assert len(results['valid_molecules']) == len(VALID_SMILES), \
            f"Expected {len(VALID_SMILES)} valid molecules, got {len(results['valid_molecules'])}"
        
        # All invalid SMILES should be caught
        invalid_count = len(results['invalid_molecules']) + len(results['sanitization_errors'])
        assert invalid_count >= len(INVALID_SMILES), "Not all invalid molecules were caught"
        
        # Properties should be calculated for all valid molecules
        assert len(results['property_calculations']) == len(results['valid_molecules']), \
            "Property calculations don't match valid molecule count"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])