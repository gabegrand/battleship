import React, { useState, useEffect, useReducer } from 'react';
import { useRound, usePlayer, useStage, useGame, useStageTimer } from "@empirica/core/player/classic/react";
import Board from '../stages/Board.js';
import { getColor } from "../../../utils/colorUtils.js";
import { SendMessage } from "../../../utils/SendMessage.js";

export function BoardComponent({init_tiles, ships}) {
  const round = useRound();
  const stage = useStage();
  const player = usePlayer();
  const game = useGame();
  const timer = useStageTimer();

  const [board, setBoard] = useState(new Board(init_tiles[0].length, init_tiles, ships));
  const [tiles, setTiles] = useState(board.getTiles());
  const [hoveredTile, setHoveredTile] = useState(null);

  useEffect(() => {
    setTiles([...board.getTiles()]);
  }, [board]);

  function isClickable(tile) {
    return tile == -1 && (stage.get("name") == "shared_stage_3" || (stage.get("name") == "instructions" && player.stage.get("textStage") == 6))
  }

  function notInInstructionsStage() {
    return !(stage.get("name") == "instructions")
  }

  function clickHandler({x, y, tile}) {
    if (isClickable(tile)) {
    if (stage.get("name") != "instructions") {
      var occTiles_new = round.get("occTiles");
      occTiles_new[x][y] = round.get("trueTiles")[x][y];
      round.set("occTiles", occTiles_new);
    } else {
      var occTiles_new = player.stage.get("occTiles");
      occTiles_new[x][y] = player.stage.get("trueTiles")[x][y];
      player.stage.set("occTiles", occTiles_new);
    }
  
      var targetLetter = String.fromCharCode("A".charCodeAt()+x);
      var targetNumber = y+1;

      var move = targetLetter.concat(targetNumber);

      var moves_new = [...round.get("moves"), move];
      round.set("moves", moves_new);

      
      if (notInInstructionsStage()) {
        if (!stage.get("answered")) {
          SendMessage(move, "move", round, game, timer);
          stage.set("answered",true);
        }
        player.stage.set("timedOut",false);
        player.stage.set("submit",true);
      } else {
        player.stage.set("textStage",player.stage.get("textStage")+1);
      }
    }
  }

  function getBrightness(isHovered, tile) {
    if (isClickable(tile) && isHovered) {
      return "brightness(75%)";
    } else {
      return "brightness(100%)";
    }
    
  }

  function handleColor(tile) {
    if (notInInstructionsStage()) {
      return getColor(tile);
    } else {
      if (player.stage.get("textStage") > 3) {
        return getColor(tile);
      }
      if (player.stage.get("textStage") > 1) {
        if ([1,2,3,4].includes(tile)) {
          return getColor(tile);
        } else {
          return getColor(-1);
        }
      }
      if (player.stage.get("textStage") > 0) {
        return getColor(-1);
      }
    }
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

  return (
    <div style={{ 
      display: 'inline-grid', 
      gridTemplateColumns: `auto repeat(${board.size}, ${cellSize}vw)`,
      gap: '0.1vw',
    }}>
      <div style={cellStyle}></div>
      {[...Array(board.size)].map((_, index) => (
        <div key={`top-${index}`} style={{
          ...cellStyle,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'black'
        }}>
          <b>{index + 1}</b>
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
            <b>{String.fromCharCode("A".charCodeAt() + x)}</b>
          </div>

          {row.map((tile, y) => (
            <div
              key={`${x}-${y}`}
              onClick={() => clickHandler({x, y, tile})}
              onMouseEnter={() => setHoveredTile(`${x}-${y}`)}
              onMouseLeave={() => setHoveredTile(null)}
              style={{
                ...cellStyle,
                border: '0.1vw solid white',
                backgroundColor: handleColor(tile),
                filter: getBrightness(hoveredTile === `${x}-${y}`, tile),
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'background-color 0.3s',
              }}
            >
            </div>
          ))}
        </React.Fragment>
      ))}
    </div>
  );
}

export default BoardComponent;