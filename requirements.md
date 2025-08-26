# SAR Model - Setup Requirements

## Create a synthetic aperture radar (SAR) Model using math formulas from reputable sources


## General Principles
- All computation is done on the Ubuntu instance
- Do not execute, download, or install packages on the MacBook
- Use SSH for all remote operations

## Prerequisites (MacBook)
- SSH key file: `/Users/mike/keys/LambdaKey.pem`
- Git repository: 
- Ubuntu instance with SSH access
- Virtual environment for safe isolation: `~/venv_sar_workspace`

## Virtual Environment Safety Setup

### Overview
To prevent any accidental writes or modifications to the MacBook system, all work should be conducted from within a dedicated virtual environment. This provides an isolated workspace while maintaining secure SSH connectivity to the Ubuntu instance for computation.

### Virtual Environment Setup (One-time)
```bash
# Create virtual environment on MacBook
python3 -m venv ~/venv_sar_workspace

# Activate virtual environment (do this every session)
source ~/venv_sar_workspace/bin/activate

# Verify activation (should show venv prefix in prompt)
# Expected: (venv_sar_workspace) (base) mike@MacBook-Pro-3 SAR %
```

### Session Startup Procedure
```bash
# 1. Navigate to project directory
cd /Users/mike/Dropbox/Code/repos/SAR

# 2. Activate virtual environment
source ~/venv_sar_workspace/bin/activate

# 3. Verify virtual environment is active
echo $VIRTUAL_ENV  # Should show: /Users/mike/venv_sar_workspace

# 4. Test Ubuntu instance connectivity
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@104.171.203.158 "echo 'Connection test successful'"
```

### Safety Benefits
- ✅ **Isolated workspace**: Prevents accidental system modifications
- ✅ **Clean environment**: No interference with MacBook system packages
- ✅ **Safe SSH operations**: All remote commands clearly separated from local system
- ✅ **Easy cleanup**: Virtual environment can be deleted if needed
- ✅ **Visual confirmation**: Prompt shows `(venv_sar_workspace)` when active

### Virtual Environment Commands
```bash
# Activate (start of each session)
source ~/venv_sar_workspace/bin/activate

# Deactivate (end of session)
deactivate

# Remove virtual environment (if needed)
rm -rf ~/venv_sar_workspace

# Recreate virtual environment (if needed)
python3 -m venv ~/venv_sar_workspace
```

## SSH Connection

### Basic Connection
```bash
# Connect to current instance
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@104.171.203.158

# For new instances, use StrictHostKeyChecking=no to avoid interactive prompts
ssh -i /Users/mike/keys/LambdaKey.pem -o StrictHostKeyChecking=no ubuntu@<INSTANCE_IP>
```

### Connection Commands (Non-Interactive)
```bash
# Execute single command
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@<INSTANCE_IP> "command"

# Chain multiple commands
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@<INSTANCE_IP> "command1 && command2"
```

### Quick Connection Workflow (From Virtual Environment)
```bash
# 0. FIRST: Activate virtual environment (REQUIRED)
source ~/venv_sar_workspace/bin/activate
# Confirm prompt shows: (venv_sar_workspace) (base) mike@MacBook-Pro-3

# 1. Verify SSH key exists (from MacBook virtual env)
ls -la /Users/mike/keys/LambdaKey.pem

# 2. Test connection and check Ubuntu environment
ssh -i /Users/mike/keys/LambdaKey.pem -o StrictHostKeyChecking=no ubuntu@104.171.203.158 "pwd && python3 --version"

# 3. Copy files to Ubuntu instance (if needed)
scp -i /Users/mike/keys/LambdaKey.pem /path/to/local/file ubuntu@104.171.203.158:~/destination/

# 4. Execute Python scripts on Ubuntu instance (ALL COMPUTATION)
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@104.171.203.158 "cd ~/project && python3 script.py"
```

### CRITICAL SAFETY REMINDER
- ⚠️  **ALWAYS activate virtual environment first**: `source ~/venv_sar_workspace/bin/activate`
- ⚠️  **NEVER run computation on MacBook**: All Python execution on Ubuntu instance only
- ⚠️  **Verify prompt**: Must show `(venv_sar_workspace)` prefix before proceeding
- ⚠️  **All file creation**: Use SSH commands to create files on Ubuntu instance

## System Specifications
- **Current Instance**: 104.171.203.158 (us-east-3)
- **Memory**: 216GB RAM
- **Storage**: 472GB + 8.0E NFS available
- **Python**: 3.10.12
- **Environment**: Jupyter, development tools pre-installed

## Chat History Maintenance Procedure

### Overview
The complete chat history must be maintained in `session_chat_history.md` to preserve all research progress, technical decisions, user inputs, and AI responses for future reference and continuation.

### Chat History Update Process

#### When to Update
- **End of each significant session** (recommended)
- **After major breakthroughs** or technical achievements
- **Before stopping Lambda instances** 
- **When user explicitly requests** ("save the chat", "save this conversation", or similar)
- **Immediately when requested** by user during active session

#### What to Include
1. **All User Inputs**: Complete user queries, questions, instructions, and comments
2. **AI Responses**: Key technical explanations, code implementations, analysis results
3. **Technical Achievements**: Successful implementations, test results, performance metrics
4. **Research Progress**: Methodology developments, algorithm improvements, validation results
5. **Infrastructure Changes**: Lambda instance migrations, system updates, configuration changes
6. **File Creations**: New scripts, datasets, analysis reports, documentation created
7. **Session Context**: Dates, instance details, system status, next steps

#### Update Format Structure
```markdown
## Session Update - [Date]: [Brief Session Description]

### [Major Topic/Achievement]
- **[Subtopic]**: Description of progress/results
- **Key Files Created**: List of new files with brief descriptions
- **Technical Results**: Performance metrics, test results, validation outcomes
- **User Inputs**: Significant user questions, decisions, or guidance provided
- **Next Steps**: Planned follow-up work or recommendations

### [Additional Topics as needed]
...

---
```

#### Step-by-Step Update Procedure

1. **Review Current Session**
   - Identify major topics covered
   - Note technical achievements and breakthroughs
   - Collect performance metrics and test results
   - Document user inputs and decisions

2. **Update session_chat_history.md**
   ```bash
   # Open the chat history file
   vim session_chat_history.md
   # OR use search_replace tool for programmatic updates
   ```

3. **Add New Session Section**
   - Add new section at the end of existing content
   - Use consistent date format: "Session Update - [Month Day, Year]"
   - Include comprehensive session summary

4. **Document User Contributions**
   - **User Questions**: Capture exact user queries that drove research direction
   - **User Decisions**: Document choices made by user (e.g., "user chose Strategy 3")
   - **User Feedback**: Include user corrections, refinements, or validation
   - **User Insights**: Capture user domain expertise or requirements clarification

5. **Include Technical Details**
   - File names and purposes of created scripts/datasets
   - Test results and performance metrics
   - Algorithm implementations and validation outcomes
   - Infrastructure changes and system configurations

6. **Commit Changes**
   ```bash
   git add session_chat_history.md
   git commit -m "Update chat history - [brief description of session]"
   ```

#### Example User Input Documentation
```markdown
### User Input Examples:

```

#### Quality Standards
- **Completeness**: Include all significant user inputs and AI responses
- **Accuracy**: Ensure technical details and results are correctly captured
- **Traceability**: Maintain clear connection between user inputs and resulting work
- **Searchability**: Use consistent terminology and section headers
- **Context**: Provide enough background for future session continuation

#### File Management
- **Primary File**: `session_chat_history.md` (main comprehensive history)
- **Backup Strategy**: Git commits provide version history
- **Size Management**: If file becomes very large (>1MB), consider archiving older sessions
- **Cross-References**: Link to specific files, test results, or documentation created

### Chat History Search and Reference

#### Finding Previous Work
```bash
# Search for specific topics
grep -n "quantum vulnerability" session_chat_history.md
grep -n "CAVP" session_chat_history.md
grep -n "User:" session_chat_history.md  # Find user inputs

# Search for specific dates or sessions
grep -n "Session Update" session_chat_history.md
```

#### Referencing in New Sessions
- Use chat history to understand previous context
- Reference user decisions and requirements from earlier sessions
- Build upon previous technical achievements
- Avoid repeating work already completed and documented

### Immediate Chat History Saving (When Requested)

#### Quick Save Process
When user requests to save the current conversation immediately:

1. **Capture Complete Conversation**
   ```bash
   # Create or update session_chat_history.md on Ubuntu instance
   ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@104.171.203.158 "cd ~ && touch session_chat_history.md"
   ```

2. **Include Full Conversation Context**
   - **Every user message**: Exact text of all user inputs, questions, and comments
   - **Every AI response**: Complete technical explanations and code implementations
   - **Conversation flow**: Maintain chronological order of exchanges
   - **Session metadata**: Date, time, instance details, current working directory

3. **Format for Complete Conversation Capture**
   ```markdown
   ## Session Update - [Month Day, Year, Time]: Complete Conversation Log
   
   ### Session Context
   - **Date/Time**: [Full timestamp]
   - **Instance**: ubuntu@104.171.203.158
   - **Working Directory**: /home/ubuntu/sar_model
   - **Session Focus**: [Brief description of main topic]
   
   ### Complete Conversation
   
   **User**: [Exact user input 1]
   
   **AI Response**: [Complete AI response 1, including any code, explanations, or actions taken]
   
   **User**: [Exact user input 2]
   
   **AI Response**: [Complete AI response 2...]
   
   [Continue for entire conversation...]
   
   ### Key Achievements This Session
   - [List major accomplishments]
   - [Files created or modified]
   - [Technical breakthroughs]
   - [User decisions made]
   
   ### Files Created/Modified
   - `filename.py`: [Brief description]
   - `output.png`: [Brief description]
   
   ### Next Steps
   - [Based on conversation context]
   
   ---
   ```

4. **Immediate Execution Commands**
   ```bash
   # Save to Ubuntu instance
   ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@104.171.203.158 "cd ~ && cat >> session_chat_history.md << 'EOF'
   [Insert formatted conversation content here]
   EOF"
   
   # Commit to git if repository exists
   ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@104.171.203.158 "cd ~ && git add session_chat_history.md && git commit -m 'Save complete chat history - [date]'"
   ```

#### Quality Requirements for Immediate Save
- **100% Completeness**: Every user comment and AI response included
- **Exact Quotations**: User inputs captured word-for-word
- **Technical Accuracy**: All code, commands, and results preserved exactly
- **Contextual Information**: Include working directories, file paths, instance details
- **Chronological Order**: Maintain exact sequence of conversation
- **Actionable Format**: Easy to reference and continue work from

#### Verification Steps
After saving chat history:
1. **Confirm file exists**: Verify session_chat_history.md was created/updated
2. **Check completeness**: Ensure all user inputs and AI responses are captured
3. **Validate format**: Confirm markdown formatting is correct
4. **Test searchability**: Verify key topics can be found with grep
5. **Backup confirmation**: Ensure git commit succeeded if applicable

## Troubleshooting

### SSH Connection Issues
1. **Host Key Verification**: New instances require host key acceptance
   - Use `-o StrictHostKeyChecking=no` for automated connection
   - Interactive prompts may timeout in tool execution

2. **Key Permissions**: Ensure proper SSH key permissions
   ```bash
   chmod 600 /Users/mike/keys/LambdaKey.pem
   ```

3. **IP Address Changes**: Update IP when instances change
   - Previous instances: 159.54.173.174, 104.171.203.47, 165.1.74.20, 129.80.141.27, 157.151.180.144 (terminated)
   - Current instance: 104.171.203.158 (us-east-3)

### Common Connection Mistakes to Avoid

1. **Running SSH from within Ubuntu instance**: 
   - ❌ WRONG: Already connected to Ubuntu, then running `ssh ubuntu@<IP>` 
   - ✅ CORRECT: Run SSH commands from MacBook terminal only
   - Check your prompt: `(base) mike@MacBook-Pro-3` = MacBook, `ubuntu@<hostname>` = Ubuntu

2. **Confusing MacBook vs Ubuntu file paths**:
   - ❌ WRONG: Looking for `/Users/mike/keys/LambdaKey.pem` on Ubuntu instance
   - ✅ CORRECT: SSH key exists on MacBook, used to connect TO Ubuntu
   - MacBook paths: `/Users/mike/...`
   - Ubuntu paths: `/home/ubuntu/...`

3. **Interactive SSH sessions in tools**:
   - ❌ WRONG: Using `ssh -i key ubuntu@ip` without commands (hangs)
   - ✅ CORRECT: Always use `ssh -i key ubuntu@ip "command"` format
   - Tool execution requires non-interactive commands

4. **Outdated IP addresses**:
   - ❌ WRONG: Using old IP from previous sessions
   - ✅ CORRECT: Always verify current instance IP before connecting
   - Update this file when instances change

### Best Practices
- Use direct command execution rather than interactive SSH sessions
- Follow `.cursorrules` SSH session management guidelines  
- Clean up temporary files before instance termination
- Always verify you're on the correct machine before running commands
- Use `scp` to transfer files from MacBook to Ubuntu instance

## Important Notes
- Always use the Ubuntu instance for computation
- SSH commands should be non-interactive for tool compatibility
- Keep instance IP addresses updated in this file
- **Maintain complete chat history** including all user inputs and AI responses
- Update chat history at end of each significant session
- Document user decisions and requirements for future reference

## Unit Testing Standards

### Overview
All SAR model implementations must include comprehensive unit tests to ensure mathematical accuracy, algorithm correctness, and system reliability. Tests should be fast enough for frequent execution but thorough enough to catch critical issues.

### Testing Framework Setup
```bash
# Install testing dependencies on Ubuntu instance
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@104.171.203.158 "pip3 install pytest numpy scipy matplotlib --user"

# Create test directory structure
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@104.171.203.158 "mkdir -p ~/sar_model/tests"
```

### Unit Test Categories

#### 1. Mathematical Validation Tests (Critical)
**Purpose**: Verify core SAR equations and algorithms
**Execution Time**: < 5 seconds
**Coverage**:
- Chirp pulse generation accuracy
- Range resolution calculations
- Phase relationships
- Matched filter responses
- Signal-to-noise ratios

```python
def test_range_resolution():
    """Test theoretical vs achieved range resolution"""
    sar = BasicSARModel(fc=10e9, B=100e6)
    expected = 1.5  # meters
    actual = sar.calculate_range_resolution()
    assert abs(actual - expected) < 0.01

def test_chirp_generation():
    """Test LFM chirp pulse properties"""
    sar = BasicSARModel()
    t, pulse = sar.generate_chirp_pulse()
    # Verify pulse length, frequency content, phase progression
    assert len(pulse) > 0
    assert np.max(np.abs(pulse)) > 0.9  # Normalized amplitude
```

#### 2. Algorithm Performance Tests (Important)
**Purpose**: Ensure processing algorithms work correctly
**Execution Time**: < 10 seconds
**Coverage**:
- Point target detection accuracy
- Range compression effectiveness
- Azimuth processing functionality
- Multi-target scenarios

```python
def test_point_target_detection():
    """Test single point target processing"""
    sar = BasicSARModel()
    t, response = sar.point_target_response(R0=1000)
    compressed = sar.range_compression(response)
    
    # Find peak location
    peak_idx = np.argmax(np.abs(compressed))
    peak_range = peak_idx * sar.c / (2 * 200e6)  # Convert to meters
    
    assert abs(peak_range - 1000) < 10  # Within 10m accuracy
```

#### 3. System Integration Tests (Moderate)
**Purpose**: Test complete processing pipeline
**Execution Time**: < 30 seconds
**Coverage**:
- End-to-end processing workflow
- File I/O operations
- Visualization generation
- Error handling

### Quick Test Execution Workflow

#### Standard Test Run (< 1 minute)
```bash
# Run core mathematical tests
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@104.171.203.158 "cd ~/sar_model && python3 -m pytest tests/test_math.py -v"

# Run algorithm tests  
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@104.171.203.158 "cd ~/sar_model && python3 -m pytest tests/test_algorithms.py -v"

# Quick integration test
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@104.171.203.158 "cd ~/sar_model && python3 -c 'from sar_basic_model import *; print(\"Quick test passed\")'"
```

#### Comprehensive Test Run (< 5 minutes)
```bash
# Run all tests with coverage
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@104.171.203.158 "cd ~/sar_model && python3 -m pytest tests/ -v --tb=short"

# Performance benchmark
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@104.171.203.158 "cd ~/sar_model && python3 tests/benchmark_performance.py"
```

### Test Data Standards

#### Synthetic Test Data (Approved for Testing)
- **Mathematical validation**: Use known analytical solutions
- **Algorithm testing**: Controlled synthetic scenarios with known outcomes
- **Performance testing**: Standardized test cases with measurable metrics

#### Test Case Examples
```python
# Standard test targets
TEST_TARGETS = [
    (0, 0, 1.0),      # Center reference target
    (100, 200, 0.8),  # Offset target for geometry testing
    (0, 1000, 0.5),   # Far-range target
]

# Standard test parameters
TEST_PARAMS = {
    'fc': 10e9,       # X-band frequency
    'B': 100e6,       # Standard bandwidth
    'PRF': 1000,      # Standard PRF
    'fs': 200e6,      # Sampling frequency
}
```

### Automated Testing Integration

#### Pre-commit Testing
```bash
# Quick validation before code changes
ssh -i /Users/mike/keys/LambdaKey.pem ubuntu@104.171.203.158 "cd ~/sar_model && python3 tests/quick_validation.py"
```

#### Continuous Integration Checks
1. **Mathematical accuracy**: All core equations within 1% tolerance
2. **Algorithm performance**: Processing time within 2x baseline
3. **Memory usage**: No memory leaks in processing pipeline
4. **Output validation**: Generated files meet format requirements

### Test Result Documentation

#### Success Criteria
- ✅ **All mathematical tests pass** (100% required)
- ✅ **Algorithm tests pass** (95% minimum)
- ✅ **Integration tests pass** (90% minimum)
- ✅ **Performance within benchmarks** (2x baseline maximum)

#### Failure Response Protocol
1. **Immediate**: Stop deployment/usage of failed component
2. **Investigation**: Identify root cause within 24 hours
3. **Fix verification**: Re-run full test suite after fixes
4. **Documentation**: Update test cases if new edge cases discovered

### Test Maintenance

#### Regular Test Updates
- **Weekly**: Run comprehensive test suite
- **Monthly**: Update test cases with new scenarios
- **Per release**: Full regression testing
- **Per instance change**: Verify environment compatibility

#### Test File Organization
```
~/sar_model/tests/
├── test_math.py              # Mathematical validation
├── test_algorithms.py        # Algorithm correctness
├── test_integration.py       # End-to-end testing
├── test_performance.py       # Speed/memory benchmarks
├── quick_validation.py       # Fast pre-commit checks
└── benchmark_performance.py  # Comprehensive benchmarking
```

## CRITICAL DATA INTEGRITY STANDARDS

### MANDATORY Data Source Requirements:

- **DO NOT USE SYNTHETIC DATA UNLESS EXPRESSLY TOLD TO DO SO**
- **Exception**: Synthetic data IS APPROVED for unit testing and algorithm validation only


### Integrity Violation Response:
- **IMMEDIATE HALT** of any training using synthetic/fake data
- **MANDATORY RE-TRAINING** with authentic data only  
- **COMPLETE AUDIT** of all ML implementations for data source compliance
- **DOCUMENTATION** of all integrity failures and corrective actions

### Prevention Measures:
- **DataIntegrityValidator** must be imported and used by ALL ML scripts
- **Automated integrity checks** before any model training begins
- **Audit trail logging** of all data validation attempts
- **Regular verification** that CSV files contain authentic data
- **Unit tests use clearly marked synthetic data for validation purposes only**
