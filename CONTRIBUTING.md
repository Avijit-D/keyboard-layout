# Contributing to Genetic Keyboard Layout Optimizer

Thank you for your interest in contributing to the Genetic Keyboard Layout Optimizer! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. **Check existing issues** first to avoid duplicates
2. **Use the issue templates** when creating new issues
3. **Provide detailed information**:
   - Python version and operating system
   - Steps to reproduce the issue
   - Expected vs actual behavior
   - Error messages and stack traces

### Suggesting Enhancements

We welcome suggestions for:
- **Cost function improvements** (better ergonomic models)
- **Genetic algorithm enhancements** (new operators, selection methods)
- **Neural network optimizations** (architecture, training)
- **Visualization improvements** (better plots, analysis tools)
- **Documentation improvements** (clarity, examples, tutorials)

### Code Contributions

#### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/genetic-keyboard-optimizer.git
   cd genetic-keyboard-optimizer
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

#### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Test your changes**:
   ```bash
   # Run the genetic algorithm with small parameters
   python genetic_keyboard_optimizer.py
   
   # Test cost calculations
   python calc_cost.py
   
   # Test neural network predictions
   python to_array.py
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## 📝 Coding Standards

### Python Style

- **Follow PEP 8** for Python code style
- **Use meaningful variable names** (e.g., `population_size` not `ps`)
- **Add docstrings** for all functions and classes
- **Keep functions focused** (single responsibility principle)
- **Use type hints** where appropriate

### Code Organization

- **Separate concerns**: Keep genetic algorithm, cost function, and neural network code separate
- **Modular design**: Make functions reusable and testable
- **Error handling**: Include proper exception handling
- **Logging**: Use print statements for user feedback, logging for debugging

### Documentation

- **Update README.md** if you add new features
- **Add docstrings** to new functions
- **Include examples** in docstrings
- **Update type hints** for new parameters

## 🧪 Testing Guidelines

### Manual Testing

Before submitting, test your changes with:

```python
# Test with small parameters first
genetic_algorithm(
    population_size=10,
    generations=5,
    plot_progress=False
)

# Test cost function
from calc_cost import layout_cost, generate_layout_from_rows
layout = generate_layout_from_rows(["qwertyuiop", "asdfghjkl", "zxcvbnm"], row_stagger_offsets)
cost = layout_cost("hello world", layout)
print(f"Cost: {cost}")

# Test neural network
from to_array import predict_cost_from_rows
predicted, actual = predict_cost_from_rows(["qwertyuiop", "asdfghjkl", "zxcvbnm"])
print(f"Predicted: {predicted}, Actual: {actual}")
```

### Performance Testing

- **Memory usage**: Monitor memory consumption during optimization
- **Speed**: Compare performance before/after changes
- **Accuracy**: Ensure neural network predictions remain accurate

## 🎯 Areas for Contribution

### High Priority

1. **Cost Function Improvements**
   - Better finger strength models
   - Hand alternation optimization
   - Typing rhythm considerations

2. **Genetic Algorithm Enhancements**
   - New crossover operators
   - Adaptive selection pressure
   - Multi-objective optimization

3. **Neural Network Improvements**
   - Architecture optimization
   - Training data augmentation
   - Model compression

### Medium Priority

1. **Visualization Enhancements**
   - Interactive plots
   - 3D keyboard visualization
   - Real-time optimization monitoring

2. **Performance Optimizations**
   - Parallel processing
   - GPU acceleration
   - Memory optimization

3. **User Experience**
   - Command-line interface
   - Configuration files
   - Progress bars and status updates

### Low Priority

1. **Documentation**
   - Tutorial notebooks
   - API documentation
   - Video tutorials

2. **Testing**
   - Unit tests
   - Integration tests
   - Performance benchmarks

## 🔍 Code Review Process

### What We Look For

- **Correctness**: Does the code work as intended?
- **Performance**: Is it efficient and scalable?
- **Readability**: Is the code clear and well-documented?
- **Compatibility**: Does it work with existing code?
- **Testing**: Has it been tested appropriately?

### Review Checklist

- [ ] Code follows Python style guidelines
- [ ] Functions have proper docstrings
- [ ] Error handling is appropriate
- [ ] Performance impact is considered
- [ ] Documentation is updated
- [ ] Changes are backward compatible

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment information**:
   - Python version
   - Operating system
   - Package versions

2. **Reproduction steps**:
   - Exact commands or code
   - Input data (if applicable)
   - Expected vs actual output

3. **Error information**:
   - Full error messages
   - Stack traces
   - Screenshots (if applicable)

## 💡 Feature Requests

When suggesting features:

1. **Describe the problem** you're trying to solve
2. **Explain your proposed solution**
3. **Provide use cases** and examples
4. **Consider implementation complexity**
5. **Think about backward compatibility**

## 📚 Resources

### Learning Materials

- [Genetic Algorithms Tutorial](https://www.tutorialspoint.com/genetic_algorithms/)
- [Neural Networks for Beginners](https://www.tensorflow.org/tutorials)
- [Python Style Guide (PEP 8)](https://www.python.org/dev/peps/pep-0008/)
- [Git Workflow Guide](https://www.atlassian.com/git/tutorials/comparing-workflows)

### Project-Specific Resources

- [Keyboard Layout Research](https://en.wikipedia.org/wiki/Keyboard_layout)
- [Ergonomic Typing Studies](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2697334/)
- [Genetic Algorithm Applications](https://www.researchgate.net/publication/220430123_Genetic_Algorithms)

## 🏆 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: [Your email] for private matters

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Genetic Keyboard Layout Optimizer! 🎉
