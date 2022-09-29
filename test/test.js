"use strict"

var fill = require("ndarray-fill")
var ops = require("ndarray-ops")
var zeros = require("zeros")

var tensordot = require("../index.js")

var test = require("tape")

test("ndarray-tensordot", function(t) {
  var x = zeros([2,3])
  var y = zeros([3,2])

  fill(x, (i,j) => x.index(i,j))
  fill(y, (i,j) => y.index(i,j))

  var add = (x,y) => x+y

  var z0 = zeros([2,3,3,2])
  fill(z0, (i,j,k,l) => x.get(i,j)*y.get(k,l))

  var z1 = zeros([2,2])
  fill(z1, (i,j) => Array.from({length:3}, (_,k) => x.get(i,k)*y.get(k,j)).reduce(add))

  var z2 = zeros([3,3])
  fill(z2, (i,j) => Array.from({length:2}, (_,k) => x.get(k,i)*y.get(j,k)).reduce(add))

  var axes0 = [[],[]]
  var axes1 = [[1],[0]]
  var axes2 = [[0],[1]]
  var axes3 = [[0,1],[1,0]]

  var oz0 = zeros([2,3,3,2])
  tensordot(oz0, x, y, axes0)
  t.assert(ops.equals(z0, oz0))

  var oz1 = zeros([2,2])
  tensordot(oz1, x, y, axes1)
  t.assert(ops.equals(z1, oz1))

  var oz2 = zeros([3,3])
  tensordot(oz2, x, y, axes2)
  t.assert(ops.equals(z2, oz2))

  var oz3 = zeros([])
  tensordot(oz3, x, y, axes3)
  //t.assert(ops.equals(z3, oz3))

  var az0 = tensordot(x, y, axes0)
  var az1 = tensordot(x, y, axes1)
  var az2 = tensordot(x, y, axes2)

  t.equals(oz0.shape.join(","), az0.shape.join(","))
  t.equals(oz1.shape.join(","), az1.shape.join(","))
  t.equals(oz2.shape.join(","), az2.shape.join(","))

  t.end()
})