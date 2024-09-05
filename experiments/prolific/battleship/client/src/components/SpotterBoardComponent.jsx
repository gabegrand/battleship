import React, { useState, useEffect } from 'react';
import { useRound, usePlayer, useStage, useGame, useStageTimer } from "@empirica/core/player/classic/react";
import Board from '../stages/Board.js';
import { getColor } from '../../../utils/colorUtils.js';

export function SpotterBoardComponent({occ_tiles, init_tiles, ships}) {
  const round = useRound();
  const stage = useStage();
  const player = usePlayer();
  const game = useGame();
  const timer = useStageTimer();

  const [board, setBoard] = useState(new Board(init_tiles[0].length, init_tiles, ships));
  const [tiles, setTiles] = useState(board.getTiles());

  useEffect(() => {
    setTiles([...board.getTiles()]);
  }, [board]);

  function notInInstructionsStage() {
    return !(stage.get("name") == "instructions")
  }

  var cellSize = 0;
  if (notInInstructionsStage()) {
    var cellSize = 20/board.size;
  } else {
    var cellSize = 15/board.size;
  }

  const cellStyle = {
    width: `${cellSize}vw`,
    height: `${cellSize}vw`,
    fontSize: `${cellSize/2}vw`,
  };

  function getHitmarker(x,y){
    const lastHits = round.get("moves");
    var currentTile = String(x)+String(y);
    var lastMoves = lastHits.map((move) => (String(move[0].charCodeAt()-"A".charCodeAt())+String(move[1]-1)));
    var whichHitmarker = lastMoves.indexOf(currentTile);
    switch (whichHitmarker) {
      case lastMoves.length-1:
        var hitmarkerOpacity = "100%";
        break;
      case lastMoves.length-2:
        if (lastMoves.length > 1) {
          var hitmarkerOpacity = "30%";
        } else {
          var hitmarkerOpacity = "0%";
        }
        break;
      default:
        var hitmarkerOpacity = "0%";
    }
    if (lastMoves.length == 0 || stage.get("name") == "instructions") {
      var hitmarkerOpacity = "0%";
    }
    return (<p style={{color:'white', fontSize:'2vw', opacity:hitmarkerOpacity}}><b>×</b></p>)
  }

  function getTileStyle(x, y) {
    const baseColor = getColor(tiles[x][y]);

    if (stage.get("name") == "instructions") {
      var isVisible = occ_tiles[x][y] === init_tiles[x][y];
    } else {
      var isVisible = round.get("occTiles")[x][y] === round.get("trueTiles")[x][y];
    }

    return {
      ...cellStyle,
      border: '0.1vw solid white',
      backgroundColor: baseColor,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      ...(isVisible ? {} : {
        backgroundImage: `
        repeating-linear-gradient(
          45deg,
          white,
          white 4px,
          ${baseColor} 4px,
          ${baseColor} 9px
        )
      `,
      backgroundSize: `${cellSize}vw ${cellSize}vw`,
      opacity:"70%"
      }),
    };
  }

  return (
    <div style={{ display: 'inline-grid', gridTemplateColumns: `auto repeat(${board.size}, ${cellSize}vw)` }}>
      <div style={cellStyle}></div> 
      {[...Array(board.size)].map((_, index) => (
        <div key={`top-${index}`} style={{
          ...cellStyle,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'black'
        }}>
          <strong>{index + 1}</strong>
        </div>
      ))}

      {tiles.map((row, x) => (
        <React.Fragment key={`row-${x}`}>
          <div style={{
            ...cellStyle,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'black'
          }}>
            <strong>{String.fromCharCode("A".charCodeAt() + x)}</strong>
          </div>

          {row.map((tile, y) => (
            <div
              key={`${x}-${y}`}
              style={getTileStyle(x, y)}
            >
              {getHitmarker(x,y)}
            </div>
          ))}
        </React.Fragment>
      ))}
    </div>
  );
}

export default SpotterBoardComponent;