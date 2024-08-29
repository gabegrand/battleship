import React, { useState, useEffect } from 'react';
import { useRound, usePlayer, useStage, useGame, useStageTimer } from "@empirica/core/player/classic/react";
import Board from './Board';

export function BoardComponent({init_tiles, ships}) {

  const round = useRound();
  const stage = useStage();
  const player = usePlayer();
  const game = useGame();
  const timer = useStageTimer();

  const [board, setBoard] = useState(new Board(3, init_tiles, ships));
  const [tiles, setTiles] = useState(board.getTiles());

  useEffect(() => {
    setTiles([...board.getTiles()]);
  }, [board]);

  function clickHandler({x, y}) {
    if (player.round.get("role") == "captain" && stage.get("name") == "shared_stage_3") {

      var occTiles_new = round.get("occTiles");
      occTiles_new[x][y] = round.get("trueTiles")[x][y];
      round.set("occTiles", occTiles_new);
  
      var targetLetter = String.fromCharCode("A".charCodeAt()+x);
      var targetNumber = y+1;

      var move = targetLetter.concat(targetNumber);

      var stage_time = game.get("roundDuration")*1000 - timer?.remaining;
            
      var newMessage = new Map();
      newMessage.set("id", round.get("messages").length);
      newMessage.set("text", move);
      newMessage.set("type", "move");
      newMessage.set("time", stage_time);

      var updated_messages = [...round.get("messages"), Object.fromEntries(newMessage)];
      round.set("messages", updated_messages);

      //sets moves array
      var moves_new = [...round.get("moves"), move];
      round.set("moves", moves_new);

      //updates move count
      round.set("score", round.get("score")+1)
  
      player.stage.set("submit",true);
    }

  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${board.size}, 50px)` }}>
      {tiles.map((row, x) =>
        row.map((tile, y) => (
          <div
            key={`${x}-${y}`}
            onClick={() => clickHandler({x, y})}
            style={{
              width: '50px',
              height: '50px',
              border: '1px solid white',
              backgroundColor: tile === 1 ? 'red' : tile === 2 ? 'blue' : tile === 0 ? 'lightgray' : 'gray',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white'
            }}
          >
            {String.fromCharCode(x+"A".charCodeAt())}{y+1}
          </div>
        ))
      )}
    </div>
  );
}

export default BoardComponent;
