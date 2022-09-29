'use strict'

var fill = require("ndarray-fill")
var pool = require('ndarray-scratch')

function range(lower, upper) {
  var length = upper-lower
  return Array.from({length}, (_, i) => lower + i)
}

function cartesianProduct(...xss) {
  return xss.reduce((a, b) => a.flatMap(d => b.map(e => [d, e].flat())), [[]])
}

function indices(shape) {
  var ranges = shape.map((n) => range(0, n))
  return cartesianProduct(...ranges)
}

function ndarrayTensordot() {
  var a, b, c, axes

  if (arguments.length === 3) {
    // With allocation (output not specified):
    a = arguments[0]
    b = arguments[1]
    axes = arguments[2]
  } else if (arguments.length === 4) {
    // Without allocation (output specified):
    c = arguments[0]
    a = arguments[1]
    b = arguments[2]
    axes = arguments[3]
  }

  function prepare(inputs) {
    var output = {
      shape: [],
      // TODO: Determine common type of inputs?
      dtype: inputs[0].dtype,
    }
    for (var k=0; k<inputs.length; k++) {
      var input = inputs[k]
      input.indexMap = []
      for (var i=0; i<input.shape.length; i++) {
        var j = input.axes.indexOf(i)
        if (j == -1) {
          input.indexMap.push([0, output.shape.length])
          output.shape.push(input.shape[i])
        } else {
          input.indexMap.push([1, j])
        }
      }
    }
    return {inputs, output}
  }

  var info = prepare([
    {axes: axes[0], shape: a.shape, dtype: a.dtype},
    {axes: axes[1], shape: b.shape, dtype: b.dtype},
  ])

  if (!c) {
    c = pool.zeros(info.output.shape, info.output.dtype)
  }

  function mapIndex(input) {
    return (index) => input.shape.map((_, k) => {
      var [i, j] = input.indexMap[k]
      return index[i][j]
    })
  }

  var indexA = mapIndex(info.inputs[0])
  var indexB = mapIndex(info.inputs[1])

  var innerShape = (k) => info.inputs[k].axes.map((i) => info.inputs[k].shape[i])

  // TODO: Check all inputs for compatible shapes?
  var innerIndices = indices(innerShape(0))

  var zero = 0
  var add = (x, y) => x + y
  var mul = (x, y) => x * y

  fill(c, (...outerIndex) => {
    return innerIndices.map((innerIndex) => {
      var index = [outerIndex, innerIndex]
      return mul(a.get(...indexA(index)), b.get(...indexB(index)))
    }).reduce(add, zero)
  })

  return c
}

module.exports = ndarrayTensordot